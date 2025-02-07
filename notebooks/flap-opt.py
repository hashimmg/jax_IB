import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax_ib.base import (
    config,
    shard_utils,
    grids,
    array_utils,
    equations,
    diffusion,
    advection,
    time_stepping,
    convolution_functions,
    IBM_Force,
)

from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from functools import partial
from jax.experimental.shard_map import shard_map

import jaxopt
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

import numpy as np
import pickle

GridVariable = grids.GridVariable


def ellipse(geometry_params, ntheta=200):
    A = geometry_params[0]
    B = geometry_params[1]
    xt = jnp.linspace(-A, A, ntheta)
    yt = B / A * jnp.sqrt(A**2 - xt**2)
    xt_2 = jnp.linspace(A, -A, ntheta)[1:-1]
    yt2 = -B / A * jnp.sqrt(A**2 - xt_2**2)
    return jnp.append(xt, xt_2), jnp.append(yt, yt2)


def initialize_params(A0, frequency, n_a, n_b, m_a, m_b):
    n_b = 20
    alpha_com = jnp.zeros(n_a).at[0].set(0.8)
    beta_com = jnp.zeros(n_b)
    alpha_rot = jnp.zeros(n_a).at[0].set(jnp.pi / 4.0)
    beta_rot = jnp.zeros(m_b)
    return alpha_com, beta_com, alpha_rot, beta_rot, 0.0


def to_motion_params(alpha_com, beta_com, alpha_rot, beta_rot, phi_rot):
    phi_dis = 0.0
    p_H = 1.0
    p_p = 1.0
    A0 = 2.0
    frequency = 1.0
    com_params = [
        A0 / 2,
        frequency,
        phi_dis,
        alpha_com * A0 / (4 * frequency),
        beta_com * A0 / (2 * frequency),
        p_H,
    ]
    rot_params = [0.0, frequency, phi_rot, alpha_rot, beta_rot, 1.0]
    return com_params, rot_params


def ellipse_trajectory(
    ellipse_parameters, initial_center_of_mass_position, motion_params, t
):
    com_params, rot_params = to_motion_params(*motion_params)
    x, y = ellipse(ellipse_parameters, 200)
    center_of_mass = initial_center_of_mass_position + com_motion(com_params, t)
    angular_rotation_speed = com_rotation(rot_params, t)
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


def ellipse_trajectory_simple(ellipse_parameters, npoints, t):
    x, y = ellipse(ellipse_parameters, npoints)
    angular_rotation_speed = 2 * jnp.pi + jnp.cos(
        2.0 * jnp.pi * t
    ) * convolution_functions.gaussian(t, 2 * jnp.pi, 1)
    xp = x * jnp.cos(angular_rotation_speed * t) - y * jnp.sin(
        angular_rotation_speed * t
    )
    yp = x * jnp.sin(angular_rotation_speed * t) + y * jnp.cos(
        angular_rotation_speed * t
    )
    return jnp.stack([xp, yp], axis=1)


def com_motion(parameters, t):
    alpha0, f, phi, alpha, beta, p = parameters
    frequencies = jnp.array([jnp.arange(1, len(alpha) + 1)])

    def alpha_1(time):
        angle = jnp.add(2 * jnp.pi * time * frequencies * f, phi)
        return jnp.sum(alpha * jnp.sin(angle) + beta * jnp.cos(angle))

    return jnp.array([-alpha0 * t, alpha_1(t) - alpha_1(0)])


def com_rotation(parameters, t):
    a, b = com_motion(parameters, t)
    return -a + b


def main(
    mesh: Mesh,
    density: float = 1,
    viscosity: float = 0.05,
    dt: float = 1e-4,
    L1: float = 30.0,
    L2: float = 10.0,
    N1: int = 128,
    N2: int = 128,
    ref_time: float = 0.0,
    inner_steps: int = 10,
    outer_steps: int = 10,
    optimize: bool = False,
):
    """
    Run an example parameter optimization.

    Args:
      mesh: The device mesh
      density: Fluid density
      viscosity: Fluid viscosity
      dt: Time step
      L1: Domain size in x direction,
      L2:Domain size in y direction
      N1: Grid shape in x direction
      N2: Grid shape in y direction
      ref_time: Reference time
      inner_steps: Inner time steps
      outer_steps: Outer time steps
      optimize: If or if not to run optimization. If False only returns the value
        of the loss at the initial parameter values.

    Returns:
      Any: The loss, or the optimal parameters
    """
    domain = ((0.0, L1), (0.0, L2))

    dtype = jnp.float64
    grid = grids.Grid(
        (N1, N2), domain=domain, device_mesh=mesh, periods=(L1, L2), dtype=dtype
    )

    # initial center of mass of the ellipse and principal axes
    initial_com = jnp.array([domain[0][1] * 0.75, domain[1][1] * 0.5])
    ellipse_params = jnp.array([0.5, 1.0])

    # initialize the pressure and velocities on each device
    dist_initialize = shard_map(
        shard_utils.dist_initialize,
        mesh=mesh,
        in_specs=(P("i", "j"), None),
        out_specs=(P("i", "j"), (P("i", "j"), P("i", "j"))),
    )

    # these are sharded now across the available devices
    global_pressure, global_velocities = dist_initialize(
        np.arange(8).reshape(4, 2), grid
    )

    surface_velocity = lambda f, x: convolution_functions.mesh_convolve(
        f, x, convolution_functions.gaussian, axis_names=("i", "j"), vmapped=False
    )

    # eigenvalues of the 1d laplacians (each spatial direction)
    # required for pseudo inversion of pressure correction
    eigvals = tuple(
        [
            np.fft.fft(array_utils.laplacian_column(size, step))
            for size, step in zip(grid.shape, grid.step)
        ]
    )

    # map the evolution function across the device mesh
    evolve_drag = shard_map(
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
        ),
        out_specs=(P("i", "j"), (P("i", "j"), P("i", "j")), (P(), P())),
        check_rep=False,
    )

    # use upwind interpolation for convection contribution
    def convect(v):
        return tuple(advection.advect_upwind(u, v, dt) for u in v)

    # function for computing explicit update steps of velocities
    explicit_update = equations.navier_stokes_explicit_terms(
        density=density,
        viscosity=viscosity,
        dt=dt,
        convect=convect,
        diffuse=diffusion.diffuse,
        forcing=None,
    )

    # the final loss function to be optimized
    def loss(params: tuple[jax.Array]):
        """
        The loss function to be optimized.

        Args:
          params: A tuple of jax.Array objects that hold the various motion parameters
            of the ellipse, i.e. rotation and translation
        """
        # ellipse_position(t) returns the surface points of the ellipse
        # at time t
        ellipse_position = partial(
            ellipse_trajectory, *[ellipse_params, initial_com, params]
        )
        d_ellipse_position = jax.jacfwd(ellipse_position)

        force_fn = lambda v, t: IBM_Force.immersed_boundary_force_per_particle(
            v, ellipse_position, convolution_functions.gaussian, surface_velocity, t, dt
        )

        def com_coords(t: float):
            """Compute the center-of-mass coordinates  of the ellipse surface points"""
            x = ellipse_position(t)
            R = jnp.mean(x, axis=0)
            return x - R

        def rotation_energy(t: float):
            """Compute the rotational energy in the center of mass reference frame of the ellipse"""
            v = jax.jacfwd(com_coords)(t)
            return jnp.sum(v**2) / 2

        # the power required for the rotation
        rotation_power = jax.grad(rotation_energy)

        #################    alternative loss function, uncomment if you wantto use it ##################
        # def compute_drag_v1(args, _):
        #     pressure, velocities, time = args

        #     x = ellipse_position(time)
        #     UP = d_ellipse_position(time)
        #     u_at_surface = jnp.stack(
        #         [surface_velocity(u, x) for u in velocities], axis=1
        #     )
        #     force = jnp.sum((UP - u_at_surface), axis=0) / dt
        #     fx, fy = force[0] * pressure.grid.step[0], force[1] * pressure.grid.step[1]
        #     Ux, Uy = jnp.mean(d_ellipse_position(time), axis=0)

        #     return (
        #         fx * Ux * dt * inner_steps,
        #         (fy * Uy + rotation_power(time)) * dt * inner_steps,
        #     )
        ################################################################################################

        # teh following function is used to compute the final loss function, i.e. the ratio of
        # energy required to move in x direction vs the energy required for lifting and rotating the ellipse.
        def compute_drag(
            args: tuple[GridVariable, tuple[GridVariable, GridVariable]], _
        ) -> tuple[jax.Array, jax.Array]:
            """
            Computes the required power for moving in x-direction,
            the power for movements in y-direction and the power for rotation.
            Returns a tuple containing the discretized integral
            of the power over the full simulation time, discretized over the
            outer loops, and the discretized time integral of the power of
            lifting and rotating the ellipse.

            """
            pressure, velocities, time = args
            x = ellipse_position(time)
            UP = d_ellipse_position(time)
            u_at_surface = jnp.stack(
                [surface_velocity(u, x) for u in velocities], axis=1
            )
            force = force_fn(velocities, t=time)
            delta = pressure.grid.step
            fx, fy = tuple(
                [
                    jax.lax.psum(
                        jax.lax.psum(
                            jnp.trapezoid(
                                jnp.trapezoid(f, axis=0, dx=delta[0]),
                                axis=0,
                                dx=delta[1],
                            ),
                            axis_name="i",
                        ),
                        axis_name="j",
                    )
                    for f in force
                ]
            )
            Ux, Uy = jnp.mean(d_ellipse_position(time), axis=0)
            return (
                fx * Ux * dt * inner_steps,
                (fy * Uy + rotation_power(time)) * dt * inner_steps,
            )

        # `y` contains the stacked outputs of `compute_drag` above.
        p, v, y = evolve_drag(
            global_pressure,
            global_velocities,
            eigvals,
            ref_time,
            dt,
            1,
            inner_steps,
            outer_steps,
            ellipse_position,
            explicit_update,
            ("i", "j"),
            None,
            compute_drag,
        )
        # compute the ration of total energy of movement in x direction
        # vs total energy of lifting and rotating the object (ellipse).
        return jnp.sum(y[0]) / jnp.sum(y[1])

    # jitted gradient of loss, returns both value of the objective as well as gradient values
    jgradloss = jax.jit(jax.value_and_grad(loss, argnums=0))

    n_a = 20
    n_b = 20
    m_a = 20
    m_b = 20
    A0 = 2.0
    frequency = 1.0
    initial_motion_params = initialize_params(A0, frequency, n_a, n_b, m_a, m_b)

    # lower and upper bounds of parameters
    lower = (
        np.concatenate([[0.25], np.full(n_a - 1, -0.8)]),
        np.full(n_b, -0.8),
        np.full(m_a, -jnp.pi / 4),
        np.full(m_b, -jnp.pi / 4),
        -jnp.pi / 2,
    )
    upper = (
        np.full(n_a, 0.8),
        np.full(n_b, 0.8),
        np.full(m_a, jnp.pi / 4),
        np.full(m_b, jnp.pi / 4),
        jnp.pi / 2,
    )
    if optimize:
        # optimize loss with jaxopt
        solver = ProjectedGradient(jgradloss, projection_box, value_and_grad=True)
        return solver.run(initial_motion_params, hyperparams_proj=(lower, upper)).params
    return jgradloss(initial_motion_params)


if __name__ == "__main__":
    mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=("i", "j"))

    density = 1.0
    viscosity = 0.003
    dt = 4e-5
    L1, L2 = 30, 10
    N1, N2 = 1400, 400
    ref_time = 0.0  # the initial time of the simulation
    inner_steps = 2  # steps of the inner loop
    outer_steps = 2  # steps of the outer loop
    optimize = False
    optimal_params = main(
        mesh,
        density,
        viscosity,
        dt,
        L1,
        L2,
        N1,
        N2,
        ref_time,
        inner_steps,
        outer_steps,
        optimize,
    )
    with open("optimal-params.pkl", "wb") as f:
        pickle.dump(optimal_params, f)
    print(optimal_params)
