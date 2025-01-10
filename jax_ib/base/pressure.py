from typing import Callable, Optional
import scipy.linalg
import numpy as np
from jax_ib.base import array_utils
#from jax_cfd.base import pressure
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

def solve_fast_diag_pinv(
    v: GridVariableVector,
    pinv: callable,
    pressure_bc: Optional[boundaries.ConstantBoundaryConditions] = None,
) -> grids.GridArray:
  """Solve for pressure using the fast diagonalization approach.

  To support backward compatibility, if the pressure_bc are not provided and
  velocity has all periodic boundaries, pressure_bc are assigned to be periodic.

  Args:
    v: a tuple of velocity values for each direction.
    q0: the starting guess for the pressure.
    pressure_bc: the boundary condition to assign to pressure. If None,
      boundary condition is infered from velocity.
    implementation: how to implement fast diagonalization.
      For non-periodic BCs will automatically be matmul.

  Returns:
    A solution to the PPE equation.
  """
  if pressure_bc is None:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  rhs = fd.divergence(v)
  rhs_transformed = _rhs_transform(rhs, pressure_bc)
  return grids.GridArray(pinv(rhs_transformed), rhs.offset, rhs.grid)


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


def projection_and_update_pressure(
    pressure: GridVariable,
    velocity: tuple[GridVariable],
    pinv: callable,
) -> GridVariableVector:
  """
  Apply pressure projection to make a velocity field divergence free.
  """
  v = velocity
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  qsol = solve_fast_diag_pinv(v, pinv)
  q = grids.GridVariable(qsol, pressure_bc)

  new_pressure_array =  grids.GridArray(qsol.data + pressure.data,qsol.offset,qsol.grid)
  new_pressure = grids.GridVariable(new_pressure_array,pressure_bc)

  q_grad = fd.forward_difference(q)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
  else:
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc).impose_bc()
        for u, q_g in zip(v, q_grad))
  return v_projected, new_pressure


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
