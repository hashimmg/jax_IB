"""Fast diagonalization method for inverting linear operators."""
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import functools
from functools import partial
import jax_ib.base.fft as fft
import jax_ib.base.fast_diagonalization as fdiag
from jax_ib.base import equations, advection, array_utils, boundaries, grids, interpolation, diffusion, pressure as prs, finite_differences as fd, IBM_Force, convolution_functions

L = 5.0
domain = ((0,L), (0,L))

def ellipse(geometry_params, ntheta=200):
  A = geometry_params[0]
  B = geometry_params[1]
  xt = jnp.linspace(-A,A,ntheta)
  yt = B/A*jnp.sqrt(A**2-xt**2)
  xt_2 = jnp.linspace(A,-A,ntheta)[1:-1]
  yt2 = -B/A*jnp.sqrt(A**2-xt_2**2)
  return jnp.append(xt,xt_2),jnp.append(yt,yt2)


def ellipse_trajectory(ellipse_parameters,
                       initial_center_of_mass_position,
                       angular_rotation_speed,
                       center_of_mass_motion_parameters, t):

  x, y = ellipse(ellipse_parameters, 200)
  amplitude, frequency = center_of_mass_motion_parameters
  center_of_mass = initial_center_of_mass_position + jnp.array([amplitude/2 * jnp.cos(2*jnp.pi*frequency*t), 0.0]) - jnp.array([amplitude/2, 0.0])
  xp = x*jnp.cos(angular_rotation_speed*t)-y*jnp.sin(angular_rotation_speed*t)+center_of_mass[0]
  yp = x*jnp.sin(angular_rotation_speed*t)+y*jnp.cos(angular_rotation_speed*t)+center_of_mass[1]
  return xp, yp

@pytest.fixture
def obj_fn():
  ellipse_params = jnp.array([0.1,0.5])
  center_position = jnp.array([L*0.75, L*0.5])
  rotation_param = jnp.array([jnp.pi/2])
  displacement_param = jnp.array([2.8,0.25])

  return partial(ellipse_trajectory, *[ellipse_params,
                                       center_position,
                                       rotation_param,
                                       displacement_param])
@pytest.fixture
def obj_fns():

  ellipse_params = jnp.array([0.1,0.5]), jnp.array([0.2,0.1])
  center_positions = jnp.array([L*0.15,L*0.2]),jnp.array([L*0.75,L*0.5])
  displacement_params = jnp.array([2.8,0.25]),jnp.array([1.2,1.25])
  rotation_params = jnp.array([jnp.pi/2]), jnp.array([jnp.pi])

  return [partial(ellipse_trajectory, *[a,b,c,d]) for a,b,c,d in zip(ellipse_params,
                                                                     center_positions,
                                                                     rotation_params,
                                                                     displacement_params)]


@pytest.fixture
def axis_names():
  return ('i','j')

@pytest.fixture
def mesh():
  return jax.make_mesh((4,2), ('i','j'))

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
             grid.eval_on_mesh(fn = lambda x, y: jnp.array(np.random.rand(*x.shape)), offset = offset), bc
          )
          for offset, bc in zip(grid.cell_faces,velocity_bc)
      ]
  )

  pressure = grids.GridVariable(
    grid.eval_on_mesh(fn = lambda x, y: jnp.cos(x+y), offset = grid.cell_center),  boundaries.get_pressure_bc_from_velocity(velocities))
  return velocities, pressure


def convect(v):
  return tuple(advection.advect_upwind(u, v, 1E-4) for u in v)



@pytest.mark.parametrize("N", [64])
def test_explicit_update(mesh, N):

  @partial(shard_map, mesh=mesh, in_specs=((P('i','j'),P('i','j')), None,None),
           out_specs=(P('i','j'),P('i','j')))
  def explicit_update_distributed(velocities, width,dt):
    i = jax.lax.axis_index('i')
    j = jax.lax.axis_index('j')
    subgrid = velocities[0].grid.subgrid((i,j), width=width)
    explicit_update = equations.navier_stokes_explicit_terms(
        density=1.0, viscosity=1.0, dt=dt,grid=subgrid, convect=convect, diffuse=diffusion.diffuse, forcing=None)

    local_velocities = tuple([u.to_subgrid((i,j), width).shard_pad(width) for u in velocities])
    return tuple([e.crop(width) for e in explicit_update(local_velocities)])

  dt=5E-4
  density=1
  viscosity=0.05

  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  velocities, pressure = setup_variables(grid)
  explicit_update_fn= equations.navier_stokes_explicit_terms(density, viscosity, dt, grid, convect, diffusion.diffuse, forcing=None)
  expected = explicit_update_fn(velocities)
  actual = explicit_update_distributed(velocities, 1, dt)
  [np.testing.assert_allclose(a.data, e.data) for a, e in zip(actual, expected)]

@pytest.mark.parametrize("N", [64])
def test_advect(mesh, N):
  @partial(shard_map, mesh=mesh, in_specs=((P('i','j'),P('i','j')), None), 
           out_specs=(P('i','j'),P('i','j')))
  def advect_distributed(velocities,width):
    i = jax.lax.axis_index('i')
    j = jax.lax.axis_index('j')

    local_velocities = tuple([u.to_subgrid((i,j), width).shard_pad(width) for u in velocities])
    result = convect(local_velocities)
    return tuple([e.crop(width) for e in convect(local_velocities)])

  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  velocities, pressure = setup_variables(grid)
  expected = convect(velocities)
  actual = advect_distributed(velocities, 1)
  [np.testing.assert_allclose(a.data, e.data) for a, e in zip(actual, expected)]



@pytest.mark.parametrize("N", [64])
def test_laplacian(mesh, N):
  @partial(shard_map, mesh=mesh, in_specs=(P('i','j'), None),
           out_specs=P('i','j'))
  def laplacian_distributed(variable,width):
      i = jax.lax.axis_index('i')
      j = jax.lax.axis_index('j')
      local_variable = variable.to_subgrid((i,j), width).shard_pad(width)
      return fd.laplacian(local_variable).crop(width)

  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  _, pressure = setup_variables(grid)
  expected = fd.laplacian(pressure)
  actual = laplacian_distributed(pressure,1)
  np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_divergence(mesh, N):
  @partial(shard_map, mesh=mesh, in_specs=((P('i','j'),P('i','j')), None),
           out_specs=P('i','j'))
  def divergence_distributed(velocities,width):
      i = jax.lax.axis_index('i')
      j = jax.lax.axis_index('j')
      local_velocities = tuple([u.to_subgrid((i,j), width).shard_pad(width) for u in velocities])
      return fd.divergence(local_velocities).crop(width)

  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  velocities, _ = setup_variables(grid)
  expected = fd.divergence(velocities)
  actual = divergence_distributed(velocities, 1)
  np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_diffuse(mesh, N, viscosity=0.05):
  @partial(shard_map, mesh=mesh, in_specs=(P('i','j'),None,None),
           out_specs=P('i','j'))
  def diffuse_distributed(variable, viscosity, width):
      i = jax.lax.axis_index('i')
      j = jax.lax.axis_index('j')
      local_variable = variable.to_subgrid((i,j), width).shard_pad(width)
      return diffusion.diffuse(local_variable, viscosity).crop(width)

  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  _, pressure = setup_variables(grid)
  expected = diffusion.diffuse(pressure, viscosity)
  actual = diffuse_distributed(pressure, viscosity,1)
  np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_linear_interpolation(mesh, N):
  @partial(shard_map, mesh=mesh, in_specs=(P('i','j'), None,None),
           out_specs=P('i','j'))
  def linear_interpolation_distributed(variable, target_offset, width):
      i = jax.lax.axis_index('i')
      j = jax.lax.axis_index('j')
      local_variable = variable.to_subgrid((i,j), width).shard_pad(width)
      return interpolation.linear(local_variable, target_offset,None,None).crop(width)

  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  _, pressure = setup_variables(grid)
  target_offsets = grids.control_volume_offsets(pressure)
  for offset in target_offsets:
    expected = interpolation.linear(pressure, offset,None,None)
    actual = linear_interpolation_distributed(pressure, offset, 1)
    np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_upwind_interpolation(mesh, N):
  @partial(shard_map, mesh=mesh, in_specs=((P('i','j'), P('i','j')), P('i','j'), None,None), 
           out_specs=P('i','j'))
  def upwind_interpolation_distributed(velocities,variable,offset,width):
    #create a subgrid for the current patch
    i = jax.lax.axis_index('i')
    j = jax.lax.axis_index('j')
    local_velocities = tuple([u.to_subgrid((i,j), width).shard_pad(width) for u in velocities])
    local_variable = variable.to_subgrid((i,j), width).shard_pad(width)
    return interpolation.upwind(local_variable, offset,local_velocities, None).crop(width)


  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  velocities, _ = setup_variables(grid)

  for v in velocities:
    target_offsets=  grids.control_volume_offsets(v)
    for offset in target_offsets:
      expected = interpolation.upwind(v, offset,velocities,None)
      actual = upwind_interpolation_distributed(velocities, v, offset, 1)
      np.testing.assert_allclose(actual.data, expected.data)

@pytest.mark.parametrize("N", [64])
def test_convolve(mesh, N, obj_fn):
  @partial(shard_map, mesh=mesh, in_specs=(P('i','j'),None, None), out_specs=P())
  def convolve_distributed(variable, obj_fn, t):
    x, y = obj_fn(t)
    i = jax.lax.axis_index('i')
    j = jax.lax.axis_index('j')

    local_variable = variable.to_subgrid((i,j), width=0)
    local_convolve = convolution_functions.mesh_convolve(
        local_variable, x, y,
        convolution_functions.gaussian, axis_names=['i','j']
    )
    return local_convolve

  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  _, pressure = setup_variables(grid)
  t = 1.0
  x,y = obj_fn(t)
  expected = convolution_functions.convolve(pressure,x,y,convolution_functions.gaussian)
  actual = convolve_distributed(pressure, obj_fn, t)
  np.testing.assert_allclose(actual, expected)

@pytest.mark.parametrize("N", [64])
def test_immersed_boundary_force(mesh, N, obj_fns):
  @partial(shard_map, mesh=mesh, in_specs=((P('i','j'), P('i','j')),None, None, None), out_specs=(P('i','j'), P('i','j')))
  def distributed_immersed_boundary_force(velocities, obj_fns, t, dt):
    i = jax.lax.axis_index('i')
    j = jax.lax.axis_index('j')

    local_velocities = tuple([u.to_subgrid((i,j), width=0) for u in velocities])
    surface_velocity = lambda f,x,y: convolution_functions.mesh_convolve(f,x,y,convolution_functions.gaussian, axis_names=['i','j'])

    forcex, forcey = IBM_Force.immersed_boundary_force(
        local_velocities, obj_fns, convolution_functions.gaussian, surface_velocity, t, dt)
    return forcex, forcey

  dt = 1E-4
  t = 1.0
  L = 5.0
  grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh = mesh, periods = (L,L))
  velocities, _ = setup_variables(grid)

  surface_velocity =  lambda field,x,y:convolution_functions.convolve(field,x,y,convolution_functions.gaussian)
  expected = IBM_Force.immersed_boundary_force(velocities, obj_fns, convolution_functions.gaussian, surface_velocity, t, dt)
  actual = distributed_immersed_boundary_force(velocities, obj_fns, t, dt)
  [np.testing.assert_allclose(a.data, e.data) for a, e in zip(actual, expected)]

