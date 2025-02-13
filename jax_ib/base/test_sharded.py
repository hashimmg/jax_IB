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
from jax_ib.base import (
    equations,
    advection,
    array_utils,
    boundaries,
    grids,
    interpolation,
    diffusion,
    pressure as prs,
    finite_differences as fd,
    IBM_Force,
    convolution_functions,
    particle_class as pc,
    time_stepping,
)
import jax_cfd

# global test variables
L = 5.0

NUM_DEVICES = len(jax.devices())


def ellipse(geometry_params, ntheta=200):
    A = geometry_params[0]
    B = geometry_params[1]
    xt = jnp.linspace(-A, A, ntheta)
    yt = B / A * jnp.sqrt(A**2 - xt**2)
    xt_2 = jnp.linspace(A, -A, ntheta)[1:-1]
    yt2 = -B / A * jnp.sqrt(A**2 - xt_2**2)
    return jnp.append(xt, xt_2), jnp.append(yt, yt2)


def ellipse_trajectory(
    ellipse_parameters,
    initial_center_of_mass_position,
    angular_rotation_speed,
    center_of_mass_motion_parameters,
    t,
):

    x, y = ellipse(ellipse_parameters, 200)
    amplitude, frequency = center_of_mass_motion_parameters
    center_of_mass = (
        initial_center_of_mass_position
        + jnp.array([amplitude / 2 * jnp.cos(2 * jnp.pi * frequency * t), 0.0])
        - jnp.array([amplitude / 2, 0.0])
    )
    xp = (
        x * jnp.cos(angular_rotation_speed * t)
        - y * jnp.sin(angular_rotation_speed * t)
        + center_of_mass[0]
    )
    yp = (
        x * jnp.sin(angular_rotation_speed * t)
        + y * jnp.cos(angular_rotation_speed * t)
        + center_of_mass[1]
    )
    return jnp.stack([xp, yp], axis=1)


@pytest.fixture
def obj_fn():
    ellipse_params = jnp.array([0.1, 0.5])
    center_position = jnp.array([L * 0.75, L * 0.5])
    rotation_param = jnp.array([jnp.pi / 2])
    displacement_param = jnp.array([2.8, 0.25])

    return partial(
        ellipse_trajectory,
        *[ellipse_params, center_position, rotation_param, displacement_param]
    )


@pytest.fixture
def obj_fns():

    ellipse_params = jnp.array([0.1, 0.5]), jnp.array([0.2, 0.1])
    center_positions = jnp.array([L * 0.15, L * 0.2]), jnp.array([L * 0.75, L * 0.5])
    displacement_params = jnp.array([2.8, 0.25]), jnp.array([1.2, 1.25])
    rotation_params = jnp.array([jnp.pi / 2]), jnp.array([jnp.pi])

    return [
        partial(ellipse_trajectory, *[a, b, c, d])
        for a, b, c, d in zip(
            ellipse_params, center_positions, rotation_params, displacement_params
        )
    ]


@pytest.fixture
def axis_names():
    return ("i", "j")


@pytest.fixture
def mesh():
    if NUM_DEVICES == 4:
        return jax.make_mesh((2, 2), ("i", "j"))
    elif NUM_DEVICES == 8:
        return jax.make_mesh((4, 2), ("i", "j"))


def setup_variables(grid):
    bc_fns = [lambda t: 0.0 for _ in range(4)]
    vx_bc = ((0.0, 0.0), (0.0, 0.0))
    vy_bc = ((0.0, 0.0), (0.0, 0.0))
    bc_fns = [lambda t: 0.0 for _ in range(4)]
    vx_bc = ((0.0, 0.0), (0.0, 0.0))
    vy_bc = ((0.0, 0.0), (0.0, 0.0))

    velocity_bc = (
        boundaries.new_periodic_boundary_conditions(
            ndim=2, bc_vals=vx_bc, bc_fn=bc_fns, time_stamp=0.0
        ),
        boundaries.new_periodic_boundary_conditions(
            ndim=2, bc_vals=vy_bc, bc_fn=bc_fns, time_stamp=0.0
        ),
    )
    vx_fn = lambda x, y: jnp.zeros_like(x)
    vy_fn = lambda x, y: jnp.zeros_like(x)
    velocities = tuple(
        [
            grids.GridVariable(
                grid.eval_on_mesh(
                    fn=lambda x, y: jnp.array(np.random.rand(*x.shape)), offset=offset
                ),
                bc,
            )
            for offset, bc in zip(grid.cell_faces, velocity_bc)
        ]
    )

    pressure = grids.GridVariable(
        grid.eval_on_mesh(fn=lambda x, y: 1 + jnp.cos(x + y), offset=grid.cell_center),
        boundaries.get_pressure_bc_from_velocity(velocities),
    )
    return velocities, pressure


def convect(v):
    return tuple(advection.advect_upwind(u, v, 1e-4) for u in v)


@pytest.mark.parametrize("N", [64])
def test_explicit_update(mesh, N):

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=((P("i", "j"), P("i", "j")), None, None),
        out_specs=(P("i", "j"), P("i", "j")),
    )
    def explicit_update_distributed(velocities, width, dt):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        explicit_update = equations.navier_stokes_explicit_terms(
            density=1.0,
            viscosity=1.0,
            dt=dt,
            convect=convect,
            diffuse=diffusion.diffuse,
            forcing=None,
        )

        local_velocities = tuple(
            [u.to_subgrid((i, j)).shard_pad(width) for u in velocities]
        )
        return tuple([e.crop(width) for e in explicit_update(local_velocities)])

    dt = 5e-4
    density = 1
    viscosity = 0.05

    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    velocities, pressure = setup_variables(grid)
    explicit_update_fn = equations.navier_stokes_explicit_terms(
        density, viscosity, dt, convect, diffusion.diffuse, forcing=None
    )
    expected = explicit_update_fn(velocities)
    actual = explicit_update_distributed(velocities, 1, dt)
    [np.testing.assert_allclose(a.data, e.data) for a, e in zip(actual, expected)]


@pytest.mark.parametrize("N", [64])
def test_advect(mesh, N):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=((P("i", "j"), P("i", "j")), None),
        out_specs=(P("i", "j"), P("i", "j")),
    )
    def advect_distributed(velocities, width):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")

        local_velocities = tuple(
            [u.to_subgrid((i, j)).shard_pad(width) for u in velocities]
        )
        result = convect(local_velocities)
        return tuple([e.crop(width) for e in convect(local_velocities)])

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    velocities, pressure = setup_variables(grid)
    expected = convect(velocities)
    actual = advect_distributed(velocities, 1)
    [np.testing.assert_allclose(a.data, e.data) for a, e in zip(actual, expected)]


@pytest.mark.parametrize("N", [64])
def test_laplacian(mesh, N):
    @partial(shard_map, mesh=mesh, in_specs=(P("i", "j"), None), out_specs=P("i", "j"))
    def laplacian_distributed(variable, width):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        local_variable = variable.to_subgrid((i, j)).shard_pad(width)
        return fd.laplacian(local_variable).crop(width)

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    _, pressure = setup_variables(grid)
    expected = fd.laplacian(pressure)
    actual = laplacian_distributed(pressure, 1)
    np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_divergence(mesh, N):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=((P("i", "j"), P("i", "j")), None),
        out_specs=P("i", "j"),
    )
    def divergence_distributed(velocities, width):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        local_velocities = tuple(
            [u.to_subgrid((i, j)).shard_pad(width) for u in velocities]
        )
        return fd.divergence(local_velocities).crop(width)

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    velocities, _ = setup_variables(grid)
    expected = fd.divergence(velocities)
    actual = divergence_distributed(velocities, 1)
    np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_diffuse(mesh, N, viscosity=0.05):
    @partial(
        shard_map, mesh=mesh, in_specs=(P("i", "j"), None, None), out_specs=P("i", "j")
    )
    def diffuse_distributed(variable, viscosity, width):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        local_variable = variable.to_subgrid((i, j)).shard_pad(width)
        return diffusion.diffuse(local_variable, viscosity).crop(width)

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    _, pressure = setup_variables(grid)
    expected = diffusion.diffuse(pressure, viscosity)
    actual = diffuse_distributed(pressure, viscosity, 1)
    np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_linear_interpolation(mesh, N):
    @partial(
        shard_map, mesh=mesh, in_specs=(P("i", "j"), None, None), out_specs=P("i", "j")
    )
    def linear_interpolation_distributed(variable, target_offset, width):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        local_variable = variable.to_subgrid((i, j)).shard_pad(width)
        return interpolation.linear(local_variable, target_offset, None, None).crop(
            width
        )

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    _, pressure = setup_variables(grid)
    target_offsets = grids.control_volume_offsets(pressure)
    for offset in target_offsets:
        expected = interpolation.linear(pressure, offset, None, None)
        actual = linear_interpolation_distributed(pressure, offset, 1)
        np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
def test_upwind_interpolation(mesh, N):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=((P("i", "j"), P("i", "j")), P("i", "j"), None, None),
        out_specs=P("i", "j"),
    )
    def upwind_interpolation_distributed(velocities, variable, offset, width):
        # create a subgrid for the current patch
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        local_velocities = tuple(
            [u.to_subgrid((i, j)).shard_pad(width) for u in velocities]
        )
        local_variable = variable.to_subgrid((i, j)).shard_pad(width)
        return interpolation.upwind(
            local_variable, offset, local_velocities, None
        ).crop(width)

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    velocities, _ = setup_variables(grid)

    for v in velocities:
        target_offsets = grids.control_volume_offsets(v)
        for offset in target_offsets:
            expected = interpolation.upwind(v, offset, velocities, None)
            actual = upwind_interpolation_distributed(velocities, v, offset, 1)
            np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("N", [64])
@pytest.mark.parametrize("vmapped", [True, False])
def test_convolve(mesh, N, obj_fn, vmapped):
    @partial(shard_map, mesh=mesh, in_specs=(P("i", "j"), None, None), out_specs=P())
    def convolve_distributed(variable, obj_fn, t):
        x = obj_fn(t)
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")

        local_variable = variable.to_subgrid((i, j))
        local_convolve = convolution_functions.mesh_convolve(
            local_variable,
            x,
            convolution_functions.gaussian,
            axis_names=["i", "j"],
            vmapped=vmapped,
        )
        return local_convolve

    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    _, pressure = setup_variables(grid)
    t = 1.0
    x = obj_fn(t)

    expected = jax.vmap(convolution_functions.convolve, in_axes=(None, 0, None))(
        pressure, x, convolution_functions.gaussian
    )
    actual = convolve_distributed(pressure, obj_fn, t)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("N", [64])
@pytest.mark.parametrize("vmapped", [True, False])
def test_immersed_boundary_force(mesh, N, obj_fns, vmapped):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=((P("i", "j"), P("i", "j")), None, None, None),
        out_specs=(P("i", "j"), P("i", "j")),
    )
    def distributed_immersed_boundary_force(velocities, obj_fns, t, dt):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")

        local_velocities = tuple([u.to_subgrid((i, j)) for u in velocities])
        surface_velocity = lambda f, x: convolution_functions.mesh_convolve(
            f, x, convolution_functions.gaussian, axis_names=["i", "j"], vmapped=vmapped
        )

        forcex, forcey = IBM_Force.immersed_boundary_force(
            local_velocities,
            obj_fns,
            convolution_functions.gaussian,
            surface_velocity,
            t,
            dt,
        )
        return forcex, forcey

    dt = 1e-4
    t = 1.0
    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    velocities, _ = setup_variables(grid)
    vconv = jax.vmap(convolution_functions.convolve, in_axes=(None, 0, None))
    surface_velocity = lambda field, x: vconv(field, x, convolution_functions.gaussian)
    expected = IBM_Force.immersed_boundary_force(
        velocities, obj_fns, convolution_functions.gaussian, surface_velocity, t, dt
    )
    actual = distributed_immersed_boundary_force(velocities, obj_fns, t, dt)
    [np.testing.assert_allclose(a.data, e.data) for a, e in zip(actual, expected)]


@pytest.mark.parametrize("N", [64])
@pytest.mark.parametrize("num_steps", [10, 1000])
@pytest.mark.parametrize("vmapped", [True, False])
def test_update_step(mesh, N, num_steps, obj_fns, vmapped):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("i", "j"),
            (P("i", "j"), P("i", "j")),
            (P("i"), P("j")),
            None,
            None,
            None,
        ),
        out_specs=(P("i", "j"), P("i", "j")),
    )
    def update_step_sharded(
        pressure, velocities, laplacian_eigenvalues, width, num_steps, dt
    ):
        i = jax.lax.axis_index("i")
        j = jax.lax.axis_index("j")
        t = num_steps * dt

        def convect(v):
            return tuple(advection.advect_upwind(u, v, dt) for u in v)

        subgrid = pressure.grid.subgrid((i, j))

        explicit_update = equations.navier_stokes_explicit_terms(
            density=1.0,
            viscosity=1.0,
            dt=5e-4,
            convect=convect,
            diffuse=diffusion.diffuse,
            forcing=None,
        )

        surface_velocity = lambda f, x: convolution_functions.mesh_convolve(
            f, x, convolution_functions.gaussian, axis_names=["i", "j"], vmapped=vmapped
        )

        cutoff = 10 * jnp.finfo(jnp.float32).eps
        eigvals = jnp.add.outer(laplacian_eigenvalues[0], laplacian_eigenvalues[1].T)
        pinv = fdiag.pseudo_poisson_inversion(
            eigvals, jnp.complex128, ("i", "j"), cutoff
        )

        local_pressure = pressure.to_subgrid((i, j)).shard_pad(width)
        local_velocities = tuple(
            [u.to_subgrid((i, j)).shard_pad(width) for u in velocities]
        )
        explicit = tuple([v.crop(width) for v in explicit_update(local_velocities)])
        dP = tuple([dp.crop(width) for dp in fd.forward_difference(local_pressure)])
        local_u_star = tuple(
            [
                u.crop(width).data + dt * e.data - dp.data
                for u, e, dp in zip(local_velocities, explicit, dP)
            ]
        )
        local_u_star = tuple(
            [
                grids.GridVariable(
                    grids.GridArray(u, os, pressure.grid.subgrid((i, j)), width=0), v.bc
                )
                for os, u, v in zip(subgrid.cell_faces, local_u_star, velocities)
            ]
        )

        forces = IBM_Force.immersed_boundary_force(
            local_u_star,
            obj_fns,
            convolution_functions.gaussian,
            surface_velocity,
            t,
            dt,
        )

        local_u_star_star = tuple(
            [u.data + dt * force.data for u, force in zip(local_u_star, forces)]
        )
        local_u_star_star = tuple(
            [
                grids.GridVariable(
                    grids.GridArray(u, offset, pressure.grid.subgrid((i, j)), width=0),
                    v.bc,
                )
                for u, v, offset in zip(
                    local_u_star_star, velocities, subgrid.cell_faces
                )
            ]
        )
        local_u_projected, local_pressure = prs.projection_and_update_pressure_sharded(
            local_pressure.crop(width), local_u_star_star, pinv, width
        )
        return local_u_projected, local_pressure

    dt = 1e-3
    L = 5.0
    grid = grids.Grid((N, N), domain=((0, L), (0, L)), device_mesh=mesh, periods=(L, L))
    velocities, pressure = setup_variables(grid)
    vconv = jax.vmap(convolution_functions.convolve, in_axes=(None, 0, None))
    surface_velocity = lambda field, x: vconv(field, x, convolution_functions.gaussian)

    def convect(v):
        return tuple(advection.advect_upwind(u, v, dt) for u in v)

    grid = pressure.grid

    explicit_update = equations.navier_stokes_explicit_terms(
        density=1.0,
        viscosity=1.0,
        dt=5e-4,
        convect=convect,
        diffuse=diffusion.diffuse,
        forcing=None,
    )
    t = num_steps * dt
    explicit = explicit_update(velocities)
    dP = fd.forward_difference(pressure)
    u_star = tuple(
        [u.data + dt * e.data - dp.data for u, e, dp in zip(velocities, explicit, dP)]
    )
    u_star = tuple(
        [
            grids.GridVariable(grids.GridArray(u, os, pressure.grid, width=0), v.bc)
            for os, u, v in zip(grid.cell_faces, u_star, velocities)
        ]
    )

    forces = IBM_Force.immersed_boundary_force(
        u_star, obj_fns, convolution_functions.gaussian, surface_velocity, t, dt
    )

    u_star_star = tuple([u.data + dt * force.data for u, force in zip(u_star, forces)])
    u_star_star = tuple(
        [
            grids.GridVariable(grids.GridArray(u, offset, pressure.grid, width=0), v.bc)
            for u, v, offset in zip(u_star_star, velocities, grid.cell_faces)
        ]
    )
    u_expected, p_expected = prs.projection_and_update_pressure(pressure, u_star_star)

    eigvals = tuple(
        [
            np.fft.fft(array_utils.laplacian_column(size, step))
            for size, step in zip(grid.shape, grid.step)
        ]
    )
    u_actual, p_actual = update_step_sharded(
        pressure, velocities, eigvals, 1, num_steps, dt
    )
    [np.testing.assert_allclose(a.data, b.data) for (a, b) in zip(u_actual, u_expected)]
    np.testing.assert_allclose(p_actual.data, p_expected.data, atol=1e-7)


@pytest.mark.parametrize("N", [64])
@pytest.mark.parametrize("inner_steps", [50])
@pytest.mark.parametrize("outer_steps", [1, 10])
@pytest.mark.parametrize("vmapped", [True, False])
def test_integration(mesh, N, inner_steps, outer_steps, obj_fn, vmapped):

    density = 1.0
    viscosity = 0.05
    dt = 1e-4

    domain = ((0.0, L), (0.0, L))
    size = (N, N)
    grid = grids.Grid(
        size, domain=domain, device_mesh=mesh, periods=(15.0, 15.0), dtype=jnp.float64
    )

    velocities, pressure = setup_variables(grid)
    all_variables = pc.All_Variables(velocities, pressure, [0], 0, [0], 0.0)

    eigvals = tuple(
        [
            np.fft.fft(array_utils.laplacian_column(size, step))
            for size, step in zip(grid.shape, grid.step)
        ]
    )

    def convect(v):
        return tuple(advection.advect_upwind(u, v, dt) for u in v)

    explicit_update = equations.navier_stokes_explicit_terms(
        density=density,
        viscosity=viscosity,
        dt=dt,
        convect=convect,
        diffuse=diffusion.diffuse,
        forcing=None,
    )

    evolve = jax.jit(
        shard_map(
            time_stepping.evolve_navier_stokes_sharded,
            mesh=mesh,
            in_specs=(
                P("i", "j"),
                (P("i", "j"), P("i", "j")),
                (P("i"), P("j")),
                P(),
                P(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            out_specs=(
                P("i", "j"),
                (P("i", "j"), P("i", "j")),
                (P(None, "i", "j"), (P(None, "i", "j"), P(None, "i", "j")), P()),
            ),
        ),
        static_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    )
    ref_time = 0.0
    p, v, y = evolve(
        pressure,
        velocities,
        eigvals,
        ref_time,
        dt,
        1,
        inner_steps,
        outer_steps,
        obj_fn,
        explicit_update,
        ("i", "j"),
        None,
        lambda args, y: args,
        False,
        False,
        False,
    )

    def internal_post_processing(all_variables, dt):
        return all_variables

    vconv = jax.vmap(convolution_functions.convolve, in_axes=(None, 0, None))
    surf_fn = lambda field, x: vconv(field, x, convolution_functions.gaussian)

    IBM_forcing = lambda velocities, t, dt: IBM_Force.immersed_boundary_force(
        velocities, [obj_fn], convolution_functions.gaussian, surf_fn, t, dt
    )
    single_step = equations.semi_implicit_navier_stokes_timeBC(
        density=density,
        viscosity=viscosity,
        dt=dt,
        grid=grid,
        convect=convect,
        pressure_solve=prs.solve_fast_diag,  # only works for periodic boundary conditions
        forcing=None,  # pfo.arbitrary_obstacle(flow_cond.pressure_gradient,perm_f),
        time_stepper=time_stepping.forward_euler_updated,  # use runge-kutta , and keep it like that
        IBM_forcing=IBM_forcing,  # compute the forcing term to update the particle
        Drag_fn=internal_post_processing,  ### TO be removed from the example
    )
    step_fn = jax_cfd.base.funcutils.repeated(single_step, steps=inner_steps)
    rollout_fn = jax_cfd.base.funcutils.trajectory(
        step_fn, outer_steps, start_with_input=True
    )
    final_result, _ = jax.device_get(rollout_fn(all_variables))
    np.testing.assert_allclose(p.data, final_result.pressure.data)
    np.testing.assert_allclose(v[0].data, final_result.velocity[0].data)
    np.testing.assert_allclose(v[1].data, final_result.velocity[1].data)
