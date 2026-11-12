"""Portable GraSS: uniform coordinate subsampling followed by signed hashing.

The projection is drawn with NumPy's PCG64 from a saved specification. It
follows the paper's sparse construction but not dattri's random stream. No
dense P-by-d matrix or concatenation of all parameter gradients is needed.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import object_hash, write_json


class GrassProjector:
    def __init__(self, parameters, dimension=16384, seed=42, chunk_elements=536870911, multiplier=4):
        if dimension < 1 or multiplier < 1 or chunk_elements < 1:
            raise ValueError("Projection sizes must be positive")
        self.parameters = [(str(name), tuple(shape)) for name, shape in parameters]
        if self.parameters != sorted(self.parameters) or len({n for n, _ in self.parameters}) != len(self.parameters):
            raise ValueError("Parameters must be unique and lexicographically sorted")
        sizes = [int(np.prod(shape)) for _, shape in self.parameters]
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("Empty parameter layout")
        self.dimension = dimension
        self.sizes = sizes
        self.offsets = np.cumsum([0] + sizes, dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(seed))
        selected, destinations, signs = [], [], []
        total = int(self.offsets[-1])
        for start in range(0, total, chunk_elements):
            size = min(chunk_elements, total-start)
            indices = np.sort(rng.choice(size, min(size, multiplier*dimension), replace=False))
            selected.append(indices + start)
            destinations.append(rng.integers(0, dimension, len(indices), dtype=np.int64))
            signs.append((rng.integers(0, 2, len(indices), dtype=np.int8)*2-1).astype(np.int8))
        self.indices = np.concatenate(selected)
        self.destinations = np.concatenate(destinations)
        self.signs = np.concatenate(signs)
        digest = hashlib.sha256()
        for array in (self.indices, self.destinations, self.signs):
            digest.update(array.tobytes())
        self.spec = {"algorithm": "grass-coordinate-hash-v1", "rng": "numpy.PCG64",
                     "dimension": dimension, "seed": seed, "multiplier": multiplier,
                     "chunk_elements": chunk_elements, "total_parameters": total,
                     "layout": self.parameters, "coordinates_sha256": digest.hexdigest(),
                     "accumulation": "float32", "rescaling": "none"}
        self.fingerprint = object_hash(self.spec)
        self._torch_cache = None

    def save(self, directory):
        directory = Path(directory)
        write_json(directory / "projection.json", {**self.spec, "fingerprint": self.fingerprint})
        with (directory / "projection.npz").open("xb") as stream:
            np.savez(stream, indices=self.indices, destinations=self.destinations, signs=self.signs)

    def project_numpy(self, gradient):
        gradient = np.asarray(gradient)
        if gradient.ndim != 1 or len(gradient) != self.offsets[-1]:
            raise ValueError("Incorrect flattened gradient shape")
        result = np.zeros(self.dimension, dtype=np.float32)
        np.add.at(result, self.destinations, gradient[self.indices].astype(np.float32)*self.signs)
        return result

    def project_parameters(self, named_parameters):
        import torch
        values = list(named_parameters)
        if [(name, tuple(p.shape)) for name, p in values] != self.parameters:
            raise ValueError("Parameter layout changed")
        device = values[0][1].device
        if self._torch_cache is None:
            cache = []
            for i, (name, _) in enumerate(values):
                left, right = np.searchsorted(self.indices, [self.offsets[i], self.offsets[i+1]])
                cache.append((torch.as_tensor(self.indices[left:right]-self.offsets[i], device=device),
                              torch.as_tensor(self.destinations[left:right], device=device),
                              torch.as_tensor(self.signs[left:right], dtype=torch.float32, device=device)))
            self._torch_cache = cache
        result = torch.zeros(self.dimension, dtype=torch.float32, device=device)
        with torch.no_grad():
            for (name, parameter), (indices, destinations, signs) in zip(values, self._torch_cache):
                if parameter.grad is None:
                    raise ValueError(f"Missing gradient for {name}; verify the parameter selection")
                if parameter.grad.device != device:
                    raise ValueError("Model sharding across devices is unsupported; use independent replicas")
                if indices.numel():
                    result.scatter_add_(0, destinations, parameter.grad.reshape(-1)[indices].float()*signs)
        return result
