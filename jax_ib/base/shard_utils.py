import jax
import jax.numpy as jnp
from jax_ib.base import boundaries, grids


def dist_initialize(x, grid):

    i = jax.lax.axis_index("i")
    j = jax.lax.axis_index("j")

    subgrid = grid.subgrid((i, j))

    def get_periodic_bc():
        return (
            (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
            (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
        )

    pressure_bc = boundaries.ConstantBoundaryConditions(
        values=((0.0, 0.0), (0.0, 0.0)),
        time_stamp=0.0,
        types=get_periodic_bc(),
        boundary_fn=[lambda t: 0.0 for _ in range(4)],
    )
    velocity_bcs = (
        boundaries.ConstantBoundaryConditions(
            values=((0.0, 0.0), (0.0, 0.0)),
            time_stamp=0.0,
            types=get_periodic_bc(),
            boundary_fn=[lambda t: 0.0 for _ in range(4)],
        ),
        boundaries.ConstantBoundaryConditions(
            values=((0.0, 0.0), (0.0, 0.0)),
            time_stamp=0.0,
            types=get_periodic_bc(),
            boundary_fn=[lambda t: 0.0 for _ in range(4)],
        ),
    )

    def to_grid(fun, offset, bc):
        return grids.GridVariable(
            grids.GridArray(fun(subgrid.shape), offset, grid, 0), bc
        )

    return to_grid(jnp.ones, grid.cell_center, pressure_bc), tuple(
        [
            to_grid(jnp.zeros, offset, bc)
            for offset, bc in zip(grid.cell_faces, velocity_bcs)
        ]
    )
