# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris._merge._HashedComponent` class."""

import dask.array as da
import numpy as np
import pytest

from iris._merge import _ArrayHashes, _HashedComponent
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord


class _SubclassedAuxCoord(AuxCoord):
    pass


def _aux(points=(1.0, 2.0), bounds=None, cls=AuxCoord, **kwargs):
    kwargs.setdefault("long_name", "foo")
    return cls(np.array(points), bounds=bounds, **kwargs)


BOUNDS = np.array([[0.5, 1.5], [1.5, 2.5]])

CASES = {
    "aux_same": (_aux(), _aux()),
    "aux_points_differ": (_aux(), _aux(points=[1.0, 3.0])),
    "aux_lazy_same": (
        _aux(points=da.arange(4.0, chunks=2)),
        _aux(points=da.arange(4.0, chunks=2)),
    ),
    "aux_lazy_differ": (
        _aux(points=da.arange(4.0, chunks=2)),
        _aux(points=da.arange(1.0, 5.0, chunks=2)),
    ),
    "aux_lazy_and_real": (_aux(points=da.arange(4.0, chunks=2)), _aux(np.arange(4.0))),
    "aux_name_differs": (_aux(), _aux(long_name="bar")),
    "aux_units_differ": (_aux(units="m"), _aux(units="km")),
    "aux_attributes_differ": (_aux(attributes={"a": 1}), _aux(attributes={"a": 2})),
    "aux_bounds_same": (_aux(bounds=BOUNDS), _aux(bounds=BOUNDS.copy())),
    "aux_bounds_differ": (_aux(bounds=BOUNDS), _aux(bounds=BOUNDS + 0.1)),
    "aux_bounds_missing": (_aux(bounds=BOUNDS), _aux()),
    "aux_shape_differs": (_aux(), _aux(points=[1.0, 2.0, 3.0])),
    "aux_negative_zero": (_aux(points=[-0.0, 1.0]), _aux(points=[0.0, 1.0])),
    "aux_nan": (_aux(points=[np.nan, 1.0]), _aux(points=[-np.nan, 1.0])),
    "aux_masked": (
        _aux(points=np.ma.masked_array([1.0, 2.0], mask=[0, 1])),
        _aux(points=np.ma.masked_array([1.0, 5.0], mask=[0, 1])),
    ),
    "aux_mask_differs": (
        _aux(points=np.ma.masked_array([1.0, 2.0], mask=[0, 1])),
        _aux(points=np.ma.masked_array([1.0, 2.0], mask=[0, 0])),
    ),
    "aux_int_and_float": (_aux(points=[1, 2]), _aux(points=[1.0, 2.0])),
    "aux_string_widths": (
        _aux(points=np.array(["a", "b"], dtype="U1")),
        _aux(points=np.array(["a", "b"], dtype="U5")),
    ),
    "aux_strings_differ": (_aux(points=["a", "b"]), _aux(points=["a", "c"])),
    "dim_same": (_aux(cls=DimCoord), _aux(cls=DimCoord)),
    "dim_circular_differs": (
        _aux(cls=DimCoord, circular=True),
        _aux(cls=DimCoord, circular=False),
    ),
    "dim_and_aux": (_aux(cls=DimCoord), _aux()),
    "aux_and_dim": (_aux(), _aux(cls=DimCoord)),
    "aux_subclass": (_aux(cls=_SubclassedAuxCoord), _aux(cls=_SubclassedAuxCoord)),
    "cell_measure_same": (
        CellMeasure([1.0, 2.0], long_name="foo"),
        CellMeasure([1.0, 2.0], long_name="foo"),
    ),
    "cell_measure_differs": (
        CellMeasure([1.0, 2.0], long_name="foo"),
        CellMeasure([1.0, 3.0], long_name="foo"),
    ),
    "cell_measure_measure_differs": (
        CellMeasure([1.0, 2.0], long_name="foo", measure="area"),
        CellMeasure([1.0, 2.0], long_name="foo", measure="volume"),
    ),
    "ancillary_variable_same": (
        AncillaryVariable(da.zeros(4, dtype=np.int8, chunks=2), long_name="foo"),
        AncillaryVariable(da.zeros(4, dtype=np.int8, chunks=2), long_name="foo"),
    ),
    "ancillary_variable_differs": (
        AncillaryVariable(da.zeros(4, dtype=np.int8, chunks=2), long_name="foo"),
        AncillaryVariable(da.ones(4, dtype=np.int8, chunks=2), long_name="foo"),
    ),
    "cell_measure_and_ancillary_variable": (
        CellMeasure([1.0, 2.0], long_name="foo"),
        AncillaryVariable([1.0, 2.0], long_name="foo"),
    ),
}


@pytest.mark.parametrize(("a", "b"), CASES.values(), ids=CASES.keys())
def test_eq(a, b):
    hashes = _ArrayHashes()
    # N.B. compare copies, as == can modify the mask of masked points.
    expected = a.copy() == b.copy()
    assert (_HashedComponent(a, hashes) == _HashedComponent(b, hashes)) == expected
    assert (_HashedComponent(b, hashes) == _HashedComponent(a, hashes)) == expected
    assert (_HashedComponent(a, hashes) != _HashedComponent(b, hashes)) != expected


def test_eq_same_component(mocker):
    a = _aux(points=da.arange(4.0, chunks=2))
    hash_array = mocker.patch("iris._merge._hash_array")
    hashes = _ArrayHashes()
    assert _HashedComponent(a, hashes) == _HashedComponent(a, hashes)
    assert hash_array.call_count == 0


def test_eq_other_type():
    a = _aux()
    assert _HashedComponent(a, _ArrayHashes()) != a


def test_name():
    assert _HashedComponent(_aux(), _ArrayHashes()).name() == "foo"
