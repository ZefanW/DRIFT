"""NumPy implementation of DRIFT's task balancing, scoring and selection.

Features must come from the same checkpoint, parameter subset and projection.
``stable`` is the paper's intercept-inclusive OLS. ``legacy-float32`` keeps the
float32 fitting convention of the paper's runs, including NumPy's default rank
cutoff.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

__all__ = ["task_centroids", "score_features", "debias_scores", "select_top_fraction"]

_FINITE_CHECK_MAX_ELEMENTS = 1_048_576


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    # Preserve a memory-mapped input; neither validation nor centroid creation
    # should materialize an entire anchor table or change its storage dtype.
    array = value if isinstance(value, np.memmap) else np.asarray(value)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    if ndim == 2:
        rows_per_chunk = max(1, _FINITE_CHECK_MAX_ELEMENTS // array.shape[1])
        for start in range(0, array.shape[0], rows_per_chunk):
            if not np.isfinite(array[start : start + rows_per_chunk]).all():
                raise ValueError(f"{name} contains non-finite values")
    elif not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _identity(value: Any, name: str) -> str | int:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError(f"{name} must be a nonempty string or an integer")


def task_centroids(
    features: np.ndarray, anchors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Average signed rollout features within query, then within task.

    Each row must correspond to exactly one anchor with ``id``, ``task``,
    ``query_id`` and reward +1 or -1. Query IDs are scoped to a task; anchor IDs
    must be globally unique. All-correct and all-incorrect queries are retained.
    Tasks are returned in lexical order. Accumulation uses float64. A zero
    centroid is retained and reported, never normalized or silently discarded.
    """
    vectors = _array(features, "features", 2)
    if len(anchors) != len(vectors):
        raise ValueError("One anchor is required per feature row")
    groups: dict[str, dict[str | int, list[int]]] = defaultdict(lambda: defaultdict(list))
    ids: set[str | int] = set()
    rewards = np.empty(len(anchors), dtype=np.float64)
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            raise ValueError(f"Anchor {index} must be a mapping")
        missing = {"id", "task", "query_id", "reward"} - anchor.keys()
        if missing:
            raise ValueError(f"Anchor {index} is missing {sorted(missing)}")
        identity = _identity(anchor["id"], "anchor id")
        if identity in ids:
            raise ValueError(f"Duplicate anchor id: {identity!r}")
        ids.add(identity)
        task = anchor["task"]
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a nonempty string")
        query = _identity(anchor["query_id"], "query_id")
        reward = anchor["reward"]
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real) or reward not in (-1, 1):
            raise ValueError(f"Anchor {identity!r} reward must be exactly +1 or -1")
        rewards[index] = reward
        groups[task][query].append(index)

    tasks = sorted(groups)
    centroids = np.empty((len(tasks), vectors.shape[1]), dtype=np.float64)
    per_task = {}
    for task_index, task in enumerate(tasks):
        queries = groups[task]
        centroid = np.zeros(vectors.shape[1], dtype=np.float64)
        all_correct = all_incorrect = mixed = 0
        for query in sorted(queries, key=lambda key: (type(key).__name__, str(key))):
            indices = queries[query]
            signs = rewards[indices]
            query_vectors = vectors[indices].astype(np.float64)
            # alpha = reward / (number of queries * rollouts for this query).
            centroid += (query_vectors * signs[:, None]).mean(axis=0) / len(queries)
            if np.all(signs == 1):
                all_correct += 1
            elif np.all(signs == -1):
                all_incorrect += 1
            else:
                mixed += 1
        if not np.isfinite(centroid).all():
            raise ValueError(f"Non-finite centroid after aggregation for task {task!r}")
        centroids[task_index] = centroid
        per_task[task] = {
            "queries": len(queries),
            "rollouts": sum(map(len, queries.values())),
            "all_correct_queries": all_correct,
            "all_incorrect_queries": all_incorrect,
            "mixed_queries": mixed,
            "absolute_weight_sum": 1.0,
            "zero_centroid": bool(np.all(centroid == 0)),
        }
    return {
        "centroids": centroids,
        "tasks": tasks,
        "diagnostics": {
            "n_anchors": len(anchors),
            "n_tasks": len(tasks),
            "per_task": per_task,
            "zero_centroid_tasks": [task for task in tasks if per_task[task]["zero_centroid"]],
        },
    }


def score_features(training: np.ndarray, centroids: np.ndarray) -> dict[str, Any]:
    """Return projected dot products [N,K] and projected L2 norms [N].

    No cosine normalization is applied. Arithmetic uses the common input dtype
    with at least float32 precision. These are norms of projected features, not
    norms of the original full-parameter gradients.
    """
    training = _array(training, "training", 2)
    centroids = _array(centroids, "centroids", 2)
    if training.shape[1] != centroids.shape[1]:
        raise ValueError("training and centroids must have the same feature dimension")
    dtype = np.result_type(training.dtype, centroids.dtype, np.float32)
    training = training.astype(dtype, copy=False)
    centroids = centroids.astype(dtype, copy=False)
    with np.errstate(over="raise", invalid="raise"):
        try:
            raw = training @ centroids.T
            norms = np.linalg.norm(training, axis=1)
        except FloatingPointError as error:
            raise ValueError("Scoring overflowed; use suitably scaled finite features") from error
    if not np.isfinite(raw).all() or not np.isfinite(norms).all():
        raise ValueError("Scoring produced non-finite values")
    return {
        "raw": raw,
        "norms": norms,
        "diagnostics": {
            "arithmetic_dtype": str(dtype),
            "zero_norm_rows": int(np.count_nonzero(norms == 0)),
            "zero_centroid_columns": np.flatnonzero(np.all(centroids == 0, axis=1)).tolist(),
        },
    }


def debias_scores(
    raw: np.ndarray,
    norms: np.ndarray,
    mode: str = "stable",
    *,
    clip_percentiles: tuple[float, float] | None = None,
    zscore: bool = False,
) -> dict[str, Any]:
    """Fit each task's log-magnitude slope and divide scores by norm**slope.

    The fitted intercept is reported but NOT subtracted from corrected scores.
    Stable mode uses float64 centered/scaled OLS with an intercept and requires
    rank two. Legacy mode calls ``np.polyfit`` on float32 logarithms with its
    default rcond and reports any rank truncation. Its correction uses float32
    division/power. It reproduces that numerical fit, not arbitrary historical
    query grouping, filtering, clipping, ordering or tied-selection behavior.

    Zero scores are excluded from the logarithmic fit and remain zero. A zero
    norm is permitted only when all corresponding raw scores are zero. An
    entirely zero task is retained with beta=0 and status ``all_zero_scores``;
    no regression is claimed for it. Nonzero tasks require at least two usable
    observations and nonconstant log norms. No epsilon pseudocount is added.

    Optional percentile clipping and population z-scoring apply only AFTER
    summing corrected task scores. Both are disabled by default; z-scoring a
    constant aggregate is rejected. ``corrected`` always remains unprocessed.
    """
    raw = _array(raw, "raw", 2)
    norms = _array(norms, "norms", 1)
    if len(raw) != len(norms):
        raise ValueError("raw and norms must have the same number of rows")
    if mode not in {"stable", "legacy-float32"}:
        raise ValueError("mode must be 'stable' or 'legacy-float32'")
    if np.any(norms < 0):
        raise ValueError("norms must be nonnegative")
    if np.any(raw[norms == 0] != 0):
        raise ValueError("A zero norm cannot have nonzero raw scores")
    dtype = np.float64 if mode == "stable" else np.float32
    with np.errstate(over="ignore", under="ignore"):
        values = raw.astype(dtype)
        magnitudes = norms.astype(dtype)
    if not np.isfinite(values).all() or not np.isfinite(magnitudes).all():
        raise ValueError(f"Inputs are not representable in {np.dtype(dtype)}")
    if np.any((raw != 0) & (values == 0)) or np.any((norms != 0) & (magnitudes == 0)):
        raise ValueError(f"Nonzero inputs underflow when cast to {np.dtype(dtype)}")

    corrected = np.zeros_like(values)
    betas = np.zeros(values.shape[1], dtype=np.float64)
    fits = []
    for column in range(values.shape[1]):
        selected = (magnitudes > 0) & (values[:, column] != 0)
        count = int(np.count_nonzero(selected))
        fit: dict[str, Any] = {
            "task_index": column,
            "n_fit": count,
            "zero_score_rows": int(np.count_nonzero(values[:, column] == 0)),
        }
        if count == 0:
            fit.update(status="all_zero_scores", rank=0, coefficients=[0.0, 0.0], singular_values=[], rcond=None)
            fits.append(fit)
            continue
        if count < 2:
            raise ValueError(f"Task {column}: at least two nonzero fit observations are required")
        x = np.log(magnitudes[selected])
        y = np.log(np.abs(values[selected, column]))
        if np.ptp(x) == 0:
            raise ValueError(f"Task {column}: log norms have no variance; slope is unidentifiable")

        if mode == "stable":
            x_mean, y_mean = x.mean(), y.mean()
            x_centered, y_centered = x - x_mean, y - y_mean
            x_scale = np.sqrt(np.mean(x_centered * x_centered))
            if not np.isfinite(x_scale) or x_scale == 0:
                raise ValueError(f"Task {column}: log-norm variance is not numerically resolvable")
            design = np.column_stack((x_centered / x_scale, np.ones(count)))
            rcond = max(design.shape) * np.finfo(np.float64).eps
            coefficients, residuals, rank, singular = np.linalg.lstsq(design, y_centered, rcond=rcond)
            if rank != 2:
                raise ValueError(f"Task {column}: stable OLS is rank deficient (rank={rank})")
            beta = coefficients[0] / x_scale
            intercept = y_mean + coefficients[1] - beta * x_mean
            fit["design"] = "float64 centered log norm / RMS, intercept; centered log magnitude"
            fit["log_norm_mean"] = float(x_mean)
            fit["log_norm_scale"] = float(x_scale)
        else:
            # full=True exposes rank/SVD/rcond without hiding a RankWarning.
            coefficients, residuals, rank, singular, rcond = np.polyfit(x, y, 1, full=True)
            beta, intercept = coefficients
            fit["design"] = "np.polyfit(float32 log norm, float32 log magnitude, 1), default rcond"

        if not np.isfinite([beta, intercept]).all():
            raise ValueError(f"Task {column}: fit produced non-finite coefficients")
        betas[column] = beta
        fit.update(
            status="ok" if rank == 2 else "rank_deficient",
            rank=int(rank),
            coefficients=[float(beta), float(intercept)],
            singular_values=singular.tolist(),
            residuals=residuals.tolist(),
            rcond=float(rcond),
        )
        fits.append(fit)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            try:
                if mode == "stable":
                    corrected[selected, column] = np.sign(values[selected, column]) * np.exp(y - beta * x)
                else:
                    denominator = np.power(magnitudes[selected], float(beta), dtype=np.float32)
                    corrected[selected, column] = values[selected, column] / denominator
            except FloatingPointError as error:
                raise ValueError(f"Task {column}: norm correction overflowed or divided by zero") from error
        if np.any(corrected[selected, column] == 0):
            raise ValueError(f"Task {column}: nonzero corrected scores underflowed to zero")

    with np.errstate(over="raise", invalid="raise"):
        try:
            scores = corrected.sum(axis=1)
        except FloatingPointError as error:
            raise ValueError("Aggregate corrected scores overflowed") from error
    if not np.isfinite(corrected).all() or not np.isfinite(scores).all():
        raise ValueError("Norm correction produced non-finite values")
    postprocess: dict[str, Any] = {"clip_percentiles": None, "zscore": bool(zscore)}
    if clip_percentiles is not None:
        bounds = np.asarray(clip_percentiles, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1] <= 100:
            raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100")
        with np.errstate(over="raise", invalid="raise"):
            try:
                low, high = np.percentile(scores, bounds)
            except FloatingPointError as error:
                raise ValueError("Percentile interpolation overflowed") from error
        if not np.isfinite([low, high]).all():
            raise ValueError("Percentile clipping produced non-finite bounds")
        scores = np.clip(scores, low, high)
        postprocess.update(clip_percentiles=bounds.tolist(), clip_values=[float(low), float(high)])
    if zscore:
        mean = float(np.mean(scores, dtype=np.float64))
        standard_deviation = float(np.std(scores, dtype=np.float64))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            raise ValueError("Cannot z-score an aggregate with zero or non-finite variance")
        scores = (scores.astype(np.float64) - mean) / standard_deviation
        postprocess.update(mean=mean, standard_deviation=standard_deviation)
    if not np.isfinite(scores).all():
        raise ValueError("Score postprocessing produced non-finite values")
    return {
        "corrected": corrected,
        "scores": scores,
        "betas": betas,
        "fits": fits,
        "diagnostics": {
            "mode": mode,
            "numpy_version": np.__version__,
            "zero_norm_rows": int(np.count_nonzero(magnitudes == 0)),
            "rank_deficient_tasks": [fit["task_index"] for fit in fits if fit["status"] == "rank_deficient"],
            "all_zero_tasks": [fit["task_index"] for fit in fits if fit["status"] == "all_zero_scores"],
            "postprocess": postprocess,
        },
    }


def select_top_fraction(scores: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Return floor(N*fraction) row indices, descending score, ties by row index.

    No clipping or normalization occurs here. A zero-sized selection is valid
    (including N<10 with the default fraction). Empty input arrays are rejected.
    """
    values = _array(scores, "scores", 1).astype(np.float64)
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and between 0 and 1")
    count = math.floor(len(values) * fraction)
    return np.argsort(-values, kind="stable")[:count]
