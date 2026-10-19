# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris._merge._ArrayHashes` class."""

import dask.array as da
import numpy as np
import pytest

from iris import _merge
from iris._merge import _ArrayHashes
from iris.tests.unit.util.test_array_equal import TEST_CASES
from iris.util import array_equal

EXTRA_CASES = [
    # Lazy arrays with the same chunks.
    (da.arange(6, chunks=2), da.arange(6, chunks=2)),
    (da.arange(6, chunks=2), da.arange(1, 7, chunks=2)),
    # Lazy arrays with different chunks, or lazy and real.
    (da.arange(6, chunks=2), da.arange(6, chunks=3)),
    (da.arange(6, chunks=2), np.arange(6)),
    # Different dtypes.
    (np.arange(3, dtype=np.int32), np.arange(3, dtype=np.float64)),
    (np.array(["a", "b"], dtype="U1"), np.array(["a", "b"], dtype="U5")),
    (np.array([1.0, 2.0], dtype="<f8"), np.array([1.0, 2.0], dtype=">f8")),
    # Values which compare equal, but have different bytes.
    (np.array([-0.0, 1.0]), np.array([0.0, 1.0])),
    (da.from_array([-0.0, 1.0], chunks=1), da.from_array([0.0, 1.0], chunks=1)),
    (np.array([np.nan, 1.0]), np.array([-np.nan, 1.0])),
    (da.from_array([np.nan, 1.0]), da.from_array([-np.nan, 1.0])),
    (
        da.ma.masked_array([-0.0, np.nan, 1.0], mask=[0, 0, 1]),
        da.ma.masked_array([0.0, -np.nan, 2.0], mask=[0, 0, 1]),
    ),
    (
        da.ma.masked_array([1.0, 2.0], mask=[0, 1]),
        da.ma.masked_array([1.0, 2.0], mask=[0, 0]),
    ),
    (da.ma.masked_array([1.0, 2.0], mask=[0, 0]), da.from_array([1.0, 2.0])),
    (da.from_array(np.array(["ab", "c"])), da.from_array(np.array(["ab", "c"]))),
    (da.from_array(np.array(["ab", "c"])), da.from_array(np.array(["ab", "d"]))),
    (np.array([1 + 0j, np.nan]), np.array([1 + 0j, np.nan])),
    (
        np.array(["NaT", "2000-01-01"], dtype="M8[s]"),
        np.array(["NaT", "2000-01-01"], dtype="M8[s]"),
    ),
    (np.array([1, "a"], dtype=object), np.array([1, "a"], dtype=object)),
    # Masked values are ignored.
    (
        np.ma.masked_array([1.0, 2.0], mask=[0, 1]),
        np.ma.masked_array([1.0, 3.0], mask=[0, 1]),
    ),
    (
        da.ma.masked_array([1.0, 2.0], mask=[0, 1]),
        da.ma.masked_array([1.0, 3.0], mask=[0, 1]),
    ),
]


def _array_cases():
    cases = [
        (a, b)
        for (a, b, _, _) in TEST_CASES
        if isinstance(a, np.ndarray | da.Array) and isinstance(b, np.ndarray | da.Array)
    ]
    # The same cases with lazy arrays, which are compared by hash.
    lazy_cases = [(da.from_array(a.copy()), da.from_array(b.copy())) for a, b in cases]
    return cases + lazy_cases + EXTRA_CASES


@pytest.mark.parametrize(("a", "b"), _array_cases())
def test_array_equal(a, b):
    hashes = _ArrayHashes()
    # N.B. array_equal can modify the mask of its first argument.
    expected = array_equal(a.copy(), b.copy(), withnans=True)
    assert hashes.array_equal(a, b) == expected
    assert hashes.array_equal(b, a) == expected


def test_array_equal_same_array():
    a = da.arange(6, chunks=2)
    assert _ArrayHashes().array_equal(a, a)


class Test_compute:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.hash_array = mocker.patch(
            "iris._merge._hash_array", wraps=_merge._hash_array
        )
        self.dask_compute = mocker.patch(
            "iris._merge.dask.compute", wraps=_merge.dask.compute
        )

    def test_hashes_once(self):
        a, b, c = [da.arange(6, chunks=2) for _ in range(3)]
        hashes = _ArrayHashes()
        hashes.compute([(a, b), (a, c)])
        assert self.hash_array.call_count == 3
        assert self.dask_compute.call_count == 1
        assert hashes.array_equal(a, b)
        assert hashes.array_equal(c, a)
        hashes.compute([(b, c)])
        assert self.hash_array.call_count == 3
        assert self.dask_compute.call_count == 1

    def test_not_comparable(self):
        a = da.arange(6, chunks=2)
        pairs = [
            (a, a),
            (a, da.arange(6, chunks=3)),
            (a, da.arange(6.0, chunks=2)),
            (a, np.arange(6)),
            (np.arange(6), np.arange(6)),
        ]
        hashes = _ArrayHashes()
        hashes.compute(pairs)
        assert self.hash_array.call_count == 0
        assert self.dask_compute.call_count == 0
        for pair in pairs:
            assert hashes.array_equal(*pair)
        assert self.hash_array.call_count == 0
