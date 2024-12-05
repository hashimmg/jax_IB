import jsonargparse as argparse
import jax
import jax.numpy as jnp
from jax_ib.base import particle_class as pc

import jax_cfd.base as cfd
import jax_ib.MD as MD
from jax import random
from jax_md import space, quantity
import jax_ib
import jax_ib.base as ib
from jax_ib.base import kinematics as ks
from jax.random import uniform as random_uniform
from jax_ib.base import grids
from jax_ib.base import boundaries
from jax_ib.base import advection,finite_differences
from jax_ib.base import IBM_Force,convolution_functions,particle_motion



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Flapping demo "
    )
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
        "--grid-shape",
        nargs="+",
        type=int,
        default=[100, 100],
        help="Shape of the underlying 2-d grid"
    )
    parser.add_argument(
        "--x-domain",
        nargs="+",
        type=float,
        default=[0, 15],
        help="Grid-domain in x-direction"
    )
    parser.add_argument(
        "--y-domain",
        nargs="+",
        type=float,
        default=[0, 15],
        help="Grid-domain in y-direction"
    )

    parser.add_argument(
        "--inner-steps",
        type=int,
        default=1000,
        help="Number of inner steps in the Navier-Stokes solver"
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=50,
        help="Number of (outer) time-steps of the Navier-Stokes solver"
    )
    parser.add_argument(
        "--viscosity",
        type=float,
        default=0.05,
        help="Fluid viscosity"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Fluid mass-density"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=5e-4,
        help="time step"
    )

    parser.add_argument("--ligand_sdf", type=str)
    parser.add_argument(
        "--datafolder",
        type=str,
        default="prion-dock-output",
        help="Folder name for storing output",
    )
    return parser

NDIMS = 2
NUM_BOUNDARIES = 2*NDIMS


if __name__== "__main__":
    parser = get_parser()
    args = parser.parse_args()
    domain=(tuple(args.x_domain),tuple(args.y_domain))
    grid = grids.Grid(tuple(args.grid_shape), domain=domain)
    bc_fns = [lambda t: 0.0 for _ in range(NUM_BOUNDARIES)] #define boundary functions
    vx_bc=((bc_fns[0](0.0), bc_fns[1](0.0)), (bc_fns[2](0.0),bc_fns[3](0.0)))
    vy_bc=((0.0, 0.0), (0, 0.0))

    # the values for bc_vals are actually violating the documented required values for periodic
    # boundary conditions; is this a problem????
    # the docstring of the function boundaries.new_periodic_boundary_conditions seems to be outdated
    # check with Mohammed if the description is still correct
    velocity_bc = (boundaries.new_periodic_boundary_conditions(ndim=NDIMS,bc_vals=vx_bc,bc_fn=bc_fns,time_stamp=0.0),
                   boundaries.new_periodic_boundary_conditions(ndim=NDIMS,bc_vals=vy_bc,bc_fn=bc_fns,time_stamp=0.0))

    vx_fn = lambda x, y: jnp.zeros_like(x)
    vy_fn = lambda x, y: jnp.zeros_like(x)


    vx_0, vy_0 = tuple(
        [
           grids.GridVariable
            (
               grid.eval_on_mesh(fn = lambda x, y: jnp.zeros_like(x), offset = offset), bc # initial values for fluid velocities are 0 both in x and y direction
            )
            for offset, bc in zip(grid.cell_faces,velocity_bc)
        ]
    )
    v0 = (vx_0, vy_0)
    # Mohammed said they're using staggered mesh, pressure in the center, velolicites at the upper and left edge of the cell-square
    # is this correct though? jax-cfd seems to be using arakawa c-grid with velocities at the face centers, and pressures on the edges?
    #-- Initial Pressure Profile
    pressure0 = grids.GridVariable(
        grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid),
        boundaries.get_pressure_bc_from_velocity((vx_0, vy_0)))

    # Immersed Objects Geometery and Initial locations
    # object is described by center of the object
    # theta is here a dummy variable; if f-term was used in addition to the f_b term, we'd need theat
    def ellipse_shape(geometry_param,theta):
        A = geometry_param[0]
        B = geometry_param[1]
        ntheta=150#30#400#51
        xt = jnp.linspace(-A,A,ntheta)
        yt = B/A*jnp.sqrt(A**2-xt**2)
        xt_2 = jnp.linspace(A,-A,ntheta)[1:-1]
        yt2 = -B/A*jnp.sqrt(A**2-xt_2**2)
        return jnp.append(xt,xt_2),jnp.append(yt,yt2)


    particle_geometry_param = jnp.array([[0.5,0.06]])
    particle_center_position = jnp.array([[domain[0][1]*0.75,domain[1][1]*0.5]]) #could be an array for multiple objects


    # Immersed Objects kinematics
    displacement_param = jnp.array([[2.8,0.25]])
    rotation_param = jnp.array([[jnp.pi/2,jnp.pi/4,0.25,0]])

    mygrids = pc.Grid1d(2, domain= (0, 2*jnp.pi)) # Only needed when using Penalty method simulation setup

    #wrap everythin into a single convenience object
    particles =  pc.particle(particle_center_position,
                             particle_geometry_param,
                             displacement_param,
                             rotation_param,mygrids,
                             ellipse_shape,
                             ks.displacement, # harmonic movement of the center in x-direction, coonstant in y-direction
                             ks.rotation) # constant rotation of the ellipse


    Intermediate_calcs = [0] ## If one is interested to perform post-calculation without the need to save large trajectories
    Step_counter = 0
    MD_state = [0] # Needed when combinting Brownian integration with the Immersed Boundary simulation

    all_variables = pc.All_Variables(particles,v0,pressure0,Intermediate_calcs,Step_counter,MD_state)


    def internal_post_processing(all_variables,dt):
        return all_variables

    # Force convolution kernel
    discrete_delta = lambda x,x0,w1: convolution_functions.delta_approx_logistjax(x,x0,w1)

    # Convultion Discretized Integral
    surf_fn =  lambda field,xp,yp:convolution_functions.new_surf_fn(field,xp,yp,discrete_delta)

    # IB forcing function
    IBM_forcing = lambda v,dt: IBM_Force.calc_IBM_force_NEW_MULTIPLE(v,discrete_delta,surf_fn,dt)

    # Update particle position function
    Update_position = particle_motion.Update_particle_position_Multiple
    
    def convect(v):
      return tuple(advection.advect_upwind(u, v, args.dt) for u in v)

    step_fn = cfd.funcutils.repeated(
        ib.equations.semi_implicit_navier_stokes_timeBC(
            density=args.density,
            viscosity=args.viscosity,
            dt=args.dt,
            grid=grid,
            convect=convect,
            pressure_solve= ib.pressure.solve_fast_diag,
            forcing=None, #pfo.arbitrary_obstacle(flow_cond.pressure_gradient,perm_f),
            time_stepper= ib.time_stepping.forward_euler_updated,
            IBM_forcing = IBM_forcing,
            Updating_Position = Update_position,
            Drag_fn = internal_post_processing, ### TO be removed from the example
            ),
        steps=args.inner_steps)


    #rollout_fn = jax.jit(cfd.funcutils.trajectory(
    #        step_fn, outer_steps, start_with_input=True))

    rollout_fn = cfd.funcutils.trajectory(
            step_fn, args.time_steps, start_with_input=True)
    final_result, trajectory = jax.device_get(rollout_fn(all_variables))

