# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks relating to :meth:`iris.cube.CubeList.merge` and ``concatenate``."""

import warnings

import dask.array as da
import numpy as np

from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube, CubeList
from iris.warnings import IrisVagueMetadataWarning

from .generate_data.stock import realistic_4d_w_everything


class Merge:
    # TODO: Improve coverage.

    cube_list: CubeList

    def setup(self):
        source_cube = realistic_4d_w_everything()

        # Merge does not yet fully support cell measures and ancillary variables.
        for cm in source_cube.cell_measures():
            source_cube.remove_cell_measure(cm)
        for av in source_cube.ancillary_variables():
            source_cube.remove_ancillary_variable(av)

        second_cube = source_cube.copy()
        scalar_coord = second_cube.coords(dimensions=[])[0]
        scalar_coord.points = scalar_coord.points + 1
        self.cube_list = CubeList([source_cube, second_cube])

    def time_merge(self):
        _ = self.cube_list.merge_cube()

    def tracemalloc_merge(self):
        _ = self.cube_list.merge_cube()

    tracemalloc_merge.number = 3  # type: ignore[attr-defined]


class MergeLazyComponents:
    """Merge cubes with large lazy aux coords, cell measures and ancillary variables.

    The cubes have one of ``n_references`` different reference coordinates (like
    a time-varying surface pressure), so they merge into that many cubes, and
    each cube is compared with several ProtoCubes.
    """

    cube_list: CubeList

    params = [[1, 6]]
    param_names = ["n_references"]

    def setup(self, n_references: int):
        n_cubes = 24
        shape = (400, 500)

        def lazy(array):
            # A unique name for each, as if loaded from separate files.
            return da.from_array(array, chunks=(200, 250), name=False)

        references = [np.full(shape, float(i)) for i in range(n_references)]
        area = np.ones(shape)
        flags = np.zeros(shape, dtype=np.int8)

        self.cube_list = CubeList()
        for i in range(n_cubes):
            cube = Cube(da.zeros(shape), long_name="foo")
            cube.add_aux_coord(DimCoord(i, long_name="level"))
            reference = lazy(references[i % n_references])
            cube.add_aux_coord(AuxCoord(reference, long_name="reference"), (0, 1))
            cube.add_cell_measure(CellMeasure(lazy(area), long_name="area"), (0, 1))
            cube.add_ancillary_variable(
                AncillaryVariable(lazy(flags), long_name="flags"), (0, 1)
            )
            self.cube_list.append(cube)

    def time_merge(self, _):
        _ = self.cube_list.merge()

    def tracemalloc_merge(self, _):
        _ = self.cube_list.merge()

    tracemalloc_merge.number = 3  # type: ignore[attr-defined]


class Concatenate:
    # TODO: Improve coverage.

    cube_list: CubeList

    params = [[False, True]]
    param_names = ["Lazy operations"]

    def setup(self, lazy_run: bool):
        warnings.filterwarnings("ignore", message="Ignoring a datum")
        warnings.filterwarnings("ignore", category=IrisVagueMetadataWarning)
        source_cube = realistic_4d_w_everything(lazy=lazy_run)
        self.cube_list = CubeList([source_cube])
        for _ in range(24):
            next_cube = self.cube_list[-1].copy()
            first_dim_coord = next_cube.coord(dimensions=0, dim_coords=True)
            first_dim_coord.points = (
                first_dim_coord.points + np.ptp(first_dim_coord.points) + 1
            )
            self.cube_list.append(next_cube)

    def time_concatenate(self, _):
        _ = self.cube_list.concatenate_cube()

    def tracemalloc_concatenate(self, _):
        _ = self.cube_list.concatenate_cube()

    tracemalloc_concatenate.number = 3  # type: ignore[attr-defined]
