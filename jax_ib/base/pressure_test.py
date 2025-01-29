"""Fast diagonalization method for inverting linear operators."""
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import functools

import jax_ib.base.fft as fft
import jax_ib.base.fast_diagonalization as fdiag
from jax_ib.base import array_utils, boundaries, grids, pressure as prs

@pytest.fixture
def axis_names():
  return ('i','j')

def setup_variables(grid):
  bc_fns = [lambda t: 0.0 for _ in range(4)]
  vx_bc=((0.0, 0.0), (0.0, 0.0))
  vy_bc=((0.0, 0.0), (0.0, 0.0))
  bc_fns = [lambda t: 0.0 for _ in range(4)]
  vx_bc=((0.0, 0.0), (0.0, 0.0))
  vy_bc=((0.0, 0.0), (0.0, 0.0))

  velocity_bc = (boundaries.new_periodic_boundary_conditions(ndim=2,bc_vals=vx_bc,bc_fn=bc_fns,time_stamp=0.0),
                 boundaries.new_periodic_boundary_conditions(ndim=2,bc_vals=vy_bc,bc_fn=bc_fns,time_stamp=0.0))
  vx_fn = lambda x, y: jnp.zeros_like(x)
  vy_fn = lambda x, y: jnp.zeros_like(x)
  velocities = tuple(
      [
         grids.GridVariable
          (
             grid.eval_on_mesh(fn = lambda x, y: jnp.sin(x + y)*jnp.cos(x-y), offset = offset), bc
          )
          for offset, bc in zip(grid.cell_faces,velocity_bc)
      ]
  )
  pressure = grids.GridVariable(
    grid.eval_on_mesh(fn = lambda x, y: jnp.cos(x+y), offset = grid.cell_center),  boundaries.get_pressure_bc_from_velocity(velocities))
  return velocities, pressure

@pytest.mark.parametrize("mesh_shape", [(1,8), (2,4), (4,2), (8,1)])
@pytest.mark.parametrize("N", [32, 64, 128])
def test_pressure_projection_sharded(N, mesh_shape, axis_names):
  L = 5.0
  mesh = jax.make_mesh(mesh_shape, axis_names)
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))

  velocities, pressure = setup_variables(grid)
  eigvals = tuple([np.fft.fft(array_utils.laplacian_column(size, step)) for size, step in zip(grid.shape, grid.step)])
  v_projected_exp, new_pressure_exp = prs.projection_and_update_pressure(pressure, velocities)
  @functools.partial(jax.jit, static_argnums = (3,))
  @functools.partial(shard_map, mesh=mesh,
           in_specs = (P('i','j'),(P('i','j'), P('i','j')), (P('i'),P('j')), None),
           out_specs = (P('i','j'),P('i','j')))
  def wrapper(pressure, velocities, laplacian_eigenvalues, width):
    i = jax.lax.axis_index('i')
    j = jax.lax.axis_index('j')
    local_velocities = tuple([u.to_subgrid((i,j), width) for u in velocities])
    local_pressure = pressure.to_subgrid((i,j), width)
    cutoff = 10 * jnp.finfo(jnp.float32).eps
    eigvals = jnp.add.outer(laplacian_eigenvalues[0], laplacian_eigenvalues[1].T)
    pinv = fdiag.pseudo_poisson_inversion(eigvals, jnp.complex128, ('i','j'), cutoff)
    v_projected, new_pressure = prs.projection_and_update_pressure_sharded(local_pressure, local_velocities, pinv,width)
    return v_projected, new_pressure

  v_projected_actual, new_pressure_actual = wrapper(pressure, velocities, eigvals, 1)
  np.testing.assert_allclose(v_projected_exp[0].data, v_projected_actual[0].data)
  np.testing.assert_allclose(v_projected_exp[1].data, v_projected_actual[1].data)
  np.testing.assert_allclose(new_pressure_exp.data, new_pressure_actual.data)
