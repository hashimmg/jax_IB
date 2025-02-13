import os
#import wandb

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

import optax
import jaxopt
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

import numpy as np
import pickle
import jsonargparse as argparse
import time

GridVariable = grids.GridVariable


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example script for CFD optimization")
    parser.add_argument(
        "--config",
        action=argparse.ActionConfigFile,
        help="Read arguments in from a yaml file. All arguments passed on the "
        "command line before the config file are overridden with values from "
        "config file (if present in config). All arguments passed after "
        "config file will override any values in config file "
        "(if present in config file).",
    )

    parser.add_argument(
        "--outer-steps",
        type=int,
        default=2,
        help="Number of outer evolution steps",
    )

    parser.add_argument(
        "--inner-steps",
        type=int,
        default=2,
        help="Number of inner evolution steps",
    )

    parser.add_argument(
        "--L1",
        type=float,
        default=10.0,
        help="Extension of the spatial domain in x direction",
    )

    parser.add_argument(
        "--L2",
        type=float,
        default=10.0,
        help="Extension of the spatial domain in x direction",
    )
    parser.add_argument(
      "--meshes",
      type=str,
      nargs='+',
    )
    parser.add_argument(
        "--N1",
        type=int,
        default=32,
        help="Gridshape in x direction",
    )

    parser.add_argument(
        "--N2",
        type=int,
        default=32,
        help="Gridshape in y direction",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--viscosity",
        type=float,
        default=0.003,
    )

    parser.add_argument(
        "--grad-type",
        type=str,
        default='jacfwd'
    )

    parser.add_argument(
        "--M",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--stepfn-cp",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--inner-cp",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--outer-cp",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="path for storing results",
    )


    return parser


def ellipse(geometry_params: tuple[float], ntheta=200) -> tuple[jax.Array]:
    """
    Return surface points (x,y) od an ellipse
    centered at the origin.

    Args:
      geometry_params: The two axes of the ellipse.

    Returns:
      The x and y coordinates of the surface points.
    """
    A = geometry_params[0]
    B = geometry_params[1]
    xt = jnp.linspace(-A, A, ntheta)
    yt = B / A * jnp.sqrt(A**2 - xt**2)
    xt_2 = jnp.linspace(A, -A, ntheta)[1:-1]
    yt2 = -B / A * jnp.sqrt(A**2 - xt_2**2)
    return jnp.append(xt, xt_2), jnp.append(yt, yt2)


def initialize_params(n_a, n_b, m_a, m_b):
    """
    Initialize the motion parameters for the ellipse; see 

    Args:
    """
    alpha_com = jnp.zeros(n_a).at[0].set(0.8)
    beta_com = jnp.zeros(n_b)
    alpha_rot = jnp.zeros(n_a).at[0].set(jnp.pi/4.0)
    beta_rot = jnp.zeros(m_b)
    return alpha_com, beta_com, alpha_rot, beta_rot, 0.0


def to_motion_params(ux, alpha_com, beta_com, alpha_rot, beta_rot, phi_rot):
    phi_com = 0.0
    theta_0 = 0.0
    phi_theta = 1.0

    com_params= [ux, phi_com,alpha_com,beta_com]
    rot_params = [theta_0, phi_theta, alpha_rot,beta_rot]
    return com_params, rot_params


def ellipse_trajectory(
    ellipse_parameters, initial_center_of_mass_position, ux, motion_params, npoints, t
):
    com_params, rot_params = to_motion_params(ux, *motion_params)
    x, y = ellipse(ellipse_parameters, npoints)
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


def ellipse_at_constant_speed(
    ellipse_parameters, initial_center_of_mass_position, ux, npoints, t
):
    x, y = ellipse(ellipse_parameters, npoints)
    center_of_mass = initial_center_of_mass_position +t*ux *jnp.array([1.0,0.0])
    xp = x + center_of_mass[0]
    yp = y + center_of_mass[1]
    return jnp.stack([xp, yp], axis=1)


def fourier_expansion(phi, alpha, beta, t):
    frequencies = jnp.arange(1,len(alpha)+1, dtype=jnp.float64)
    angles = 2*jnp.pi*t*frequencies + phi
    return jnp.sum(alpha*jnp.sin(angles) + beta*jnp.cos(angles))


def com_motion(parameters,t):
    ux, phi, alpha, beta = parameters
    y = partial(fourier_expansion, phi, alpha, beta)
    return jnp.array([ux * t, y(t)- y(0.0)])


def com_rotation(parameters, t):
    theta_0, phi, alpha, beta = parameters
    theta = partial(fourier_expansion, phi, alpha, beta)
    return theta_0 + theta(t)


def main(
    mesh: Mesh,
    density: float = 1.0,
    viscosity: float = 0.05,
    ux: float = -1.0,
    dt: float = 1e-5,
    L1: float = 30.0,
    L2: float = 10.0,
    N1: int = 128,
    N2: int = 128,
    ref_time: float = 0.0,
    inner_steps: int = 10,
    outer_steps: int = 10,
    npoints: int = 100,
    M: int = 4,
    A: int=0.5,
    B: int=1.0,
    stepfn_cp=True,
    inner_cp=False,
    outer_cp=False,
    grad_type='jacfwd',
):
    """
    Run an example parameter optimization.

    Args:
      mesh: The device mesh
      density: Fluid density
      viscosity: Fluid viscosity
      ux: speed of movement of ellipse in x direction
      dt: Time step
      L1: Domain size in x direction,
      L2:Domain size in y direction
      N1: Grid shape in x direction
      N2: Grid shape in y direction
      ref_time: Reference time
      inner_steps: Inner time steps
      outer_steps: Outer time steps
      M: Number of motion parameters for center-of-mass motion and rotation,
        respectively, see optimal swimmer chapter in https://arxiv.org/abs/2403.06257
      A, B: the two axes of the ellipse.

    Returns:
      Any: The loss, or the optimal parameters
    """
    #wandb.init(project = 'aramco-jax-cfd-opt')
    domain = ((0.0, L1), (0.0, L2))

    dtype = jnp.float64
    grid = grids.Grid(
        (N1, N2), domain=domain, device_mesh=mesh, periods=(L1, L2), dtype=dtype
    )

    # initial center of mass of the ellipse and principal axes
    initial_com = jnp.array([domain[0][1] * 0.75, domain[1][1] * 0.5])
    ellipse_params = jnp.array([A, B])

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
            ellipse_trajectory, *[ellipse_params, initial_com, ux, params, npoints]
        )
        d_ellipse_position = jax.jacfwd(ellipse_position)

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


        # the following function is used to compute the final loss function, i.e. the ratio of
        # energy required to move in x direction vs the energy required for lifting and rotating the ellipse.
        def compute_drag(
            args: tuple[GridVariable, tuple[GridVariable, GridVariable], tuple[GridVariable, GridVariable], float], _
        ) -> tuple[jax.Array, jax.Array]:
            """
            Computes the required power for moving in x-direction,
            the power for movements in y-direction and the power for rotation.
            Returns a tuple containing the discretized integral
            of the power over the full simulation time, discretized over the
            outer loops, and the discretized time integral of the power of
            lifting and rotating the ellipse.

            """
            pressure, _, force, t = args
            x = ellipse_position(t)
            delta = pressure.grid.step
            fx, fy = tuple(
              [
                jax.lax.psum(
                  jax.lax.psum(
                    jnp.sum(f)*delta[0] * delta[1],
                    axis_name="i"),
                  axis_name="j")
                for f in force
              ]
            )
            Ux, Uy = jnp.mean(d_ellipse_position(t), axis=0)
            return (
                fx * Ux * dt * inner_steps,
                (fy * Uy + rotation_power(t)) * dt * inner_steps,
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
            stepfn_cp,
            inner_cp,
            outer_cp,
        )
        # compute the ration of total energy of movement in x direction
        # vs total energy of lifting and rotating the object (ellipse).
        return jnp.sum(y[0]) / jnp.sum(y[1])


    initial_motion_params = initialize_params(M, M,M,M)
    # lower and upper bounds of parameters
    jloss = jax.jit(loss)
    if grad_type == "value_and_grad":
      jgradloss = jax.jit(jax.value_and_grad(loss, argnums=0))
    elif grad_type == "grad":
      jgradloss = jax.jit(jax.grad(loss, argnums=0))
    elif grad_type == "jacrev":
      jgradloss = jax.jit(jax.jacrev(loss, argnums=0))
    elif grad_type == "jacfwd":
      jgradloss = jax.jit(jax.jacfwd(loss, argnums=0))


    jgradloss(initial_motion_params)
    res = jloss(initial_motion_params)
    print(res)


    t1 = time.time()
    res=jloss(initial_motion_params)
    res.block_until_ready()
    loss_time = time.time() - t1

    t1 = time.time()
    res = jgradloss(initial_motion_params)
    res[0].block_until_ready()
    grad_time = time.time() - t1

    return loss_time, grad_time


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    path = f"{args.path}-{args.grad_type}"
    if not os.path.exists(path):
      os.mkdir(path)

    parser.save(args, os.path.join(path, 'config.yml'))
    meshes = [tuple(list(map(int, s.split(',')))) for s in args.meshes]
    runtimes = {}

    for mesh_size in meshes:
      mesh = jax.make_mesh(axis_shapes=mesh_size, axis_names=("i", "j"))
      loss_time, grad_time = main(
        mesh=mesh,
        density=1.0,
        viscosity=args.viscosity,
        ux=-1.0,
        dt=args.dt,
        L1=args.L1,
        L2=args.L2,
        N1=args.N1,
        N2=args.N2,
        ref_time=0.0,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        npoints=100,
        M=args.M,
        A = 0.5,
        B = 1.0,
        stepfn_cp = args.stepfn_cp,
        inner_cp = args.inner_cp,
        outer_cp = args.outer_cp,
        grad_type=args.grad_type,
      )
      runtimes[mesh_size] = {'loss':loss_time,
                        'grad':grad_time}
    with open(os.path.join(path, 'runtimes.pkl'),'wb') as f:
        pickle.dump(runtimes, f)
