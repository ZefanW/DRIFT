"""CPU tests for weighting, norm correction and deterministic selection."""

import unittest
from unittest import mock
from pathlib import Path
import tempfile

import numpy as np

from drift.scoring import debias_scores, score_features, select_top_fraction, task_centroids


class TaskCentroidTests(unittest.TestCase):
    def test_memmap_uses_bounded_validation_and_only_query_sized_casts(self):
        cast_shapes = []
        finite_shapes = []
        full_shape = (6, 4)

        class GuardedMemmap(np.memmap):
            def astype(self, *args, **kwargs):
                cast_shapes.append(self.shape)
                if self.shape == full_shape:
                    raise AssertionError("Centroid aggregation must not cast the full anchor table")
                return super().astype(*args, **kwargs)

        original_isfinite = np.isfinite

        def record_isfinite(value, *args, **kwargs):
            if getattr(value, "ndim", 0) == 2:
                finite_shapes.append(value.shape)
            return original_isfinite(value, *args, **kwargs)

        anchors = [
            {"id": index, "task": "t", "query_id": query, "reward": reward}
            for index, (query, reward) in enumerate(
                [("a", 1), ("b", -1), ("b", 1), ("c", -1), ("c", -1), ("c", -1)]
            )
        ]
        values = np.arange(24, dtype=np.float32).reshape(full_shape)
        expected = task_centroids(values, anchors)
        with tempfile.TemporaryDirectory() as directory:
            mapped = GuardedMemmap(Path(directory) / "anchors.bin", dtype=np.float32, mode="w+", shape=full_shape)
            mapped[:] = values
            with mock.patch("drift.scoring._FINITE_CHECK_MAX_ELEMENTS", 8), mock.patch(
                "drift.scoring.np.isfinite", side_effect=record_isfinite
            ):
                actual = task_centroids(mapped, anchors)
            self.assertEqual(mapped.dtype, np.float32)
            del mapped
        np.testing.assert_array_equal(actual["centroids"], expected["centroids"])
        self.assertEqual(actual["diagnostics"], expected["diagnostics"])
        self.assertEqual(finite_shapes, [(2, 4), (2, 4), (2, 4)])
        self.assertEqual(cast_shapes, [(1, 4), (2, 4), (3, 4)])

    def test_query_balance_retains_all_correct_and_all_incorrect_queries(self):
        features = np.array([[2, 0], [4, 0], [0, 8], [9, 3]], dtype=float)
        anchors = [
            {"id": "a", "task": "A", "query_id": "q", "reward": 1},
            {"id": "b", "task": "A", "query_id": "q", "reward": 1},
            {"id": "c", "task": "A", "query_id": "r", "reward": -1},
            {"id": "d", "task": "B", "query_id": "q", "reward": 1},
        ]
        result = task_centroids(features, anchors)
        np.testing.assert_allclose(result["centroids"], [[1.5, -4], [9, 3]])
        self.assertEqual(result["tasks"], ["A", "B"])
        self.assertEqual(result["diagnostics"]["per_task"]["A"]["all_correct_queries"], 1)
        self.assertEqual(result["diagnostics"]["per_task"]["A"]["all_incorrect_queries"], 1)
        self.assertEqual(result["diagnostics"]["per_task"]["A"]["absolute_weight_sum"], 1)
        # Duplicating every rollout of one query cannot increase its task weight.
        extra = [{**anchor, "id": anchor["id"] + "copy"} for anchor in anchors[:2]]
        duplicated = task_centroids(np.vstack((features, features[:2])), anchors + extra)
        np.testing.assert_allclose(duplicated["centroids"], result["centroids"])

    def test_zero_centroid_is_reported_and_retained(self):
        anchors = [
            {"id": 0, "task": "t", "query_id": 0, "reward": 1},
            {"id": 1, "task": "t", "query_id": 0, "reward": -1},
        ]
        result = task_centroids([[3, 4], [3, 4]], anchors)
        np.testing.assert_array_equal(result["centroids"], [[0, 0]])
        self.assertEqual(result["diagnostics"]["zero_centroid_tasks"], ["t"])
        self.assertEqual(result["diagnostics"]["per_task"]["t"]["mixed_queries"], 1)

    def test_invalid_anchors_and_features_are_rejected(self):
        anchor = {"id": "x", "task": "t", "query_id": "q", "reward": 1}
        cases = [
            ([[1], [2]], [anchor, anchor]),
            ([[1]], [{**anchor, "reward": 0}]),
            ([[1]], [{**anchor, "reward": True}]),
            ([[np.nan]], [anchor]),
            ([[1]], []),
            (np.empty((0, 1)), []),
            ([[1]], [{**anchor, "query_id": None}]),
        ]
        for features, anchors in cases:
            with self.subTest(features=features, anchors=anchors), self.assertRaises(ValueError):
                task_centroids(features, anchors)


class FeatureScoreTests(unittest.TestCase):
    def test_dot_products_and_projected_norms_are_not_cosine(self):
        result = score_features(np.array([[3, 4], [0, 0]], dtype=np.float32), [[1, 0], [0, 2]])
        np.testing.assert_allclose(result["raw"], [[3, 8], [0, 0]])
        np.testing.assert_allclose(result["norms"], [5, 0])
        self.assertEqual(result["diagnostics"]["zero_norm_rows"], 1)

    def test_invalid_dimensions_and_nonfinite_features(self):
        with self.assertRaises(ValueError):
            score_features([[1, 2]], [[1]])
        with self.assertRaises(ValueError):
            score_features([[np.inf]], [[1]])


class DebiasTests(unittest.TestCase):
    def test_power_law_keeps_sign_and_fitted_intercept_scale(self):
        norms = np.array([1, 2, 4, 8, 16.0])
        signs = np.array([1, -1, 1, -1, 1])
        raw = np.column_stack((3 * signs * norms**2, -7 * norms**0.5))
        result = debias_scores(raw, norms)
        np.testing.assert_allclose(result["betas"], [2, 0.5], atol=1e-13)
        np.testing.assert_allclose(result["corrected"], np.column_stack((3 * signs, np.full(5, -7))), atol=1e-12)
        np.testing.assert_allclose(result["scores"], 3 * signs - 7, atol=1e-12)
        np.testing.assert_allclose([f["coefficients"][1] for f in result["fits"]], np.log([3, 7]))
        self.assertTrue(all(f["rank"] == 2 for f in result["fits"]))
        self.assertEqual(result["diagnostics"]["postprocess"]["clip_percentiles"], None)
        self.assertFalse(result["diagnostics"]["postprocess"]["zscore"])

    def test_stable_centering_handles_narrow_log_norm_range(self):
        x = np.linspace(10, 10.0001, 512)
        result = debias_scores(np.exp(1.7 * x + 2)[:, None], np.exp(x))
        self.assertAlmostEqual(result["betas"][0], 1.7, places=8)
        self.assertEqual(result["fits"][0]["rank"], 2)

    def test_zero_rows_and_zero_scores_have_explicit_diagnostics(self):
        norms = np.array([0, 1, 2, 4, 8.0])
        raw = np.column_stack(([0, 3, 0, 48, -192], np.zeros(5)))
        result = debias_scores(raw, norms)
        np.testing.assert_allclose(result["corrected"][:, 0], [0, 3, 0, 3, -3])
        self.assertEqual(result["diagnostics"]["zero_norm_rows"], 1)
        self.assertEqual(result["diagnostics"]["all_zero_tasks"], [1])
        self.assertEqual(result["fits"][1]["rank"], 0)
        self.assertEqual(result["fits"][0]["n_fit"], 3)

    def test_legacy_matches_float32_polyfit_and_exposes_rank_truncation(self):
        norms = np.exp(np.linspace(10, 10.0001, 1024)).astype(np.float32)
        raw = np.power(norms, 1.5, dtype=np.float32)[:, None]
        x, y = np.log(norms), np.log(raw[:, 0])
        coefficients, _, rank, singular, rcond = np.polyfit(x, y, 1, full=True)
        result = debias_scores(raw, norms, "legacy-float32")
        self.assertEqual(rank, 1)  # Small deterministic example of the old cutoff issue.
        self.assertEqual(result["fits"][0]["status"], "rank_deficient")
        self.assertEqual(result["fits"][0]["rank"], rank)
        self.assertEqual(result["diagnostics"]["rank_deficient_tasks"], [0])
        np.testing.assert_array_equal(result["fits"][0]["coefficients"], coefficients)
        np.testing.assert_array_equal(result["fits"][0]["singular_values"], singular)
        self.assertEqual(result["fits"][0]["rcond"], rcond)
        self.assertEqual(rcond, len(x) * np.finfo(np.float32).eps)
        expected = raw[:, 0] / np.power(norms, float(coefficients[0]), dtype=np.float32)
        np.testing.assert_array_equal(result["corrected"][:, 0], expected)
        stable = debias_scores(raw, norms)
        self.assertEqual(stable["fits"][0]["rank"], 2)
        self.assertGreater(abs(result["betas"][0] - stable["betas"][0]), 0.5)

    def test_invalid_fit_inputs_fail_instead_of_becoming_zero_scores(self):
        cases = [
            ([[1], [2]], [1, 1]),
            ([[1], [0]], [1, 2]),
            ([[1], [2]], [0, 2]),
            ([[1], [2]], [-1, 2]),
            ([[1], [np.nan]], [1, 2]),
            ([[1], [2]], [1, np.inf]),
        ]
        for raw, norms in cases:
            for mode in ["stable", "legacy-float32"]:
                with self.subTest(raw=raw, norms=norms, mode=mode), self.assertRaises(ValueError):
                    debias_scores(raw, norms, mode)
        with self.assertRaises(ValueError):
            debias_scores([[1], [2]], [1, 2], "unknown")

    def test_postprocessing_is_explicit_and_occurs_after_task_sum(self):
        norms = np.arange(1, 9, dtype=float)
        signs = np.array([1, 1, 1, -1, 1, -1, 1, -1])
        raw = np.column_stack((norms**1.5 * signs, norms * np.array([1, 2, 1, 3, 1, 2, 1, 8])))
        unprocessed = debias_scores(raw, norms)
        result = debias_scores(raw, norms, clip_percentiles=(10, 90), zscore=True)
        expected = np.clip(unprocessed["scores"], *np.percentile(unprocessed["scores"], [10, 90]))
        expected = (expected - expected.mean()) / expected.std()
        np.testing.assert_allclose(result["scores"], expected)
        np.testing.assert_array_equal(result["corrected"], unprocessed["corrected"])
        with self.assertRaises(ValueError):
            debias_scores(raw, norms, clip_percentiles=(90, 10))
        with self.assertRaises(ValueError):
            debias_scores(np.zeros((8, 1)), norms, zscore=True)
        with self.assertRaises(ValueError):
            debias_scores([[1e308], [-1e308], [1e308], [-1e308]], [1, 2, 3, 4], clip_percentiles=(40, 60))


class SelectionTests(unittest.TestCase):
    def test_floor_count_and_deterministic_tie_breaking(self):
        scores = np.array([4, 2, 4, -1, 4, 3.0])
        np.testing.assert_array_equal(select_top_fraction(scores, 0.5), [0, 2, 4])
        np.testing.assert_array_equal(select_top_fraction(scores, 0.4), [0, 2])
        np.testing.assert_array_equal(select_top_fraction(scores, 1), [0, 2, 4, 5, 1, 3])
        self.assertEqual(len(select_top_fraction(scores)), 0)
        self.assertEqual(len(select_top_fraction(scores, 0)), 0)

    def test_invalid_selection_is_rejected(self):
        for fraction in [-0.1, 1.1, np.nan, True]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                select_top_fraction([1, 2], fraction)
        with self.assertRaises(ValueError):
            select_top_fraction([1, np.inf])


if __name__ == "__main__":
    unittest.main()
