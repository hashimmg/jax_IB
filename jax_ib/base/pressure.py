from typing import Callable, Optional
import scipy.linalg
import numpy as np
from jax_ib.base import array_utils
from jax_ib.base import fast_diagonalization
import jax.numpy as jnp
from jax_ib.base import grids
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_ib.base import particle_class

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def solve_linear(local_velocities: tuple[GridVariable],
                 pinv: callable, width:int)->grids.GridArray:
  """
  Invert  ∇²p = ∇u for p on distributed hardware.

  Args:
    local_velocities: the local shards of the global velocity field
    pinv: the (sharded/distributed) function for inverting the above equation.
    width: padding width for local arrays used when computing local
      finite-differences

  Returns:
    GridArray: The solution
  """
  pressure_bc = boundaries.get_pressure_bc_from_velocity(local_velocities)
  local_velocities = tuple([v.grow(width) for v in local_velocities])
  rhs =  fd.divergence(local_velocities).crop(width)
  return grids.GridArray(pinv(rhs.data), rhs.offset, rhs.grid, rhs.width)


def projection_and_update_pressure_sharded(
    pressure: grids.GridVariable,
    velocities: tuple[grids.GridVariable], pinv:callable, width:int
) -> tuple[tuple[GridVariable, GridVariable], GridVariable]:
  """
  Sharded (distributed) version of pressure projection. 
  """
  pressure_bc = boundaries.get_pressure_bc_from_velocity(velocities)
  solution = grids.GridVariable(solve_linear(velocities, pinv, width), pressure_bc)
  new_pressure_array =  grids.GridArray(solution.data + pressure.data,pressure.offset,pressure.grid, pressure.width)
  new_pressure = grids.GridVariable(new_pressure_array,pressure_bc)

  grads =  tuple([g.crop(width) for g in fd.forward_difference(solution.grow(width))])
  grads = tuple([grids.GridVariable(g, solution.bc) for g in grads])

  v_projected = tuple(grids.GridVariable(u.array - g.array, u.bc) for u, g in zip(velocities, grads))
  return v_projected, new_pressure


def _rhs_transform(
    u: grids.GridArray,
    bc: boundaries.BoundaryConditions,
) -> Array:
  """Transform the RHS of pressure projection equation for stability.

  In case of poisson equation, the kernel is subtracted from RHS for stability.

  Args:
    u: a GridArray that solves ∇²x = u.
    bc: specifies boundary of x.

  Returns:
    u' s.t. u = u' + kernel of the laplacian.
  """
  u_data = u.data
  for axis in range(u.grid.ndim):
    if bc.types[axis][0] == boundaries.BCType.NEUMANN and bc.types[axis][
        1] == boundaries.BCType.NEUMANN:
      # if all sides are neumann, poisson solution has a kernel of constant
      # functions. We substact the mean to ensure consistency.
      u_data = u_data - jnp.mean(u_data)
  return u_data


def solve_fast_diag(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None) -> GridArray:
  """
  Solve for pressure using the fast diagonalization approach.
  This version is less general than the one in jax-cfd and works
  only for periodic boundary conditions.
  """
  del q0  # unused
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic velocity BC')
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
  return grids.applied(pinv)(rhs)


def projection_and_update_pressure(
    pressure: GridVariable,
    velocity: tuple[GridVariable],
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  v = velocity
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid, pressure.width)
  q0 = grids.GridVariable(q0, pressure_bc)

  qsol = solve(v, q0)
  q = grids.GridVariable(qsol, pressure_bc)

  New_pressure_Array =  grids.GridArray(qsol.data + pressure.data,qsol.offset,qsol.grid, pressure.width)
  New_pressure = grids.GridVariable(New_pressure_Array,pressure_bc)

  q_grad = fd.forward_difference(q)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
  else:
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc).impose_bc()
        for u, q_g in zip(v, q_grad))
  return v_projected, New_pressure


def solve_fast_diag_moving_wall(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul') -> GridArray:
  """Solve for channel flow pressure using fast diagonalization."""
  del q0  # unused
  ndim = len(v)

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = [
      array_utils.laplacian_matrix(grid.shape[0], grid.step[0]),
      array_utils.laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  ]
  for d in range(2, ndim):
    laplacians += [array_utils.laplacian_matrix(grid.shape[d], grid.step[d])]
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs)


def solve_fast_diag_Far_Field(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None) -> GridArray:
  """Solve for pressure using the fast diagonalization approach."""
  del q0  # unused

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  rhs_transformed = _rhs_transform(rhs, pressure_bc)
  #laplacians = [
  #          laplacian_matrix_neumann(grid.shape[0], grid.step[0]),
  #          laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  #]
  laplacians = array_utils.laplacian_matrix_w_boundaries(
      rhs.grid, rhs.offset, pressure_bc)
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs_transformed.dtype,
      hermitian=True, circulant=False, implementation='matmul')
  return grids.applied(pinv)(rhs)
