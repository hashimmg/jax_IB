import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax_ib.base.grids import GridVariable, GridArrayVector, GridArray
from jax_ib.base.particle_class import Particle


def immersed_boundary_force_per_particle(
    velocity_field: tuple[GridVariable, GridVariable],
    particle: callable,
    dirac_delta_approx: callable,
    surface_velocity: callable,
    t: float,
    dt: float,
) -> jax.Array:
    """
    Compute the x and y forces from an immersed object. The object is represented as a point-cloud, as returned by `shape_fn`.
    The 2-d velocity field is given by a GridArrayVector `velocity_field`.  `geom_param` and `Grid_p` are parameters passed
    to `shape_fn` required to compute the point cloud of the object.

    TODO: the use of `shape_fn` could and should be generalized.

    Args:
      velocity_field: the velocity field, i.e. vx or vy component
      particle: Callable computing the particle geometry at time `t`. Signature is `x, y = particle(t)`. The particle
        is represented as a point cloud `x, y`. The function has to be jax differentiable w.r.t. `t`.
      dirac_delta_approx: Approximation to the delta function
      surface_velocity: Callable which computes the surface-integral `sum_{i,j} data[i,j] delta(x[i]-xp]) delta(y[j]-yp)  dx  dy)`
        for field.data
      t: The time.
      dt: The time step.

    Returns:
      jax.Array of shape grid.shape: The forces Fx and Fy acting on the velocity fields vx and vy due to the presence of the object
        shape_fn (represented as a point cloud)
    """
    ux, uy = velocity_field
    X, Y = ux.grid.mesh(ux.offset)  # 2d is hard coded right now
    dx = ux.grid.step[0]

    xp, yp = particle(t)
    UPx, UPy = jax.jacfwd(particle)(t)

    ux_at_surface = surface_velocity(ux, xp, yp)
    uy_at_surface = surface_velocity(uy, xp, yp)

    forcex = (UPx - ux_at_surface) / dt
    forcey = (UPy - uy_at_surface) / dt

    x_i = jnp.roll(xp, -1)
    y_i = jnp.roll(yp, -1)
    dxL = x_i - xp
    dyL = y_i - yp
    dS = jnp.sqrt(dxL**2 + dyL**2)

    def calc_force(F, xp, yp, dss):
        return (
            F * dirac_delta_approx(jnp.sqrt((xp - X) ** 2 + (yp - Y) ** 2), 0, dx) * dss
        )
        # return F*dirac_delta_approx(xp-X,0,dx)*dirac_delta_approx(yp-Y,0,dy)*dss
        # return F*dirac_delta_approx(xp,X,dx)*dirac_delta_approx(yp,Y,dy)*dss**2

    vmapped_calc_force = jax.vmap(calc_force, in_axes=0)
    # TODO (mganahl): the two vmap calls can be done in parallel as well
    return jnp.sum(vmapped_calc_force(forcex, xp, yp, dS), axis=0), jnp.sum(
        vmapped_calc_force(forcey, xp, yp, dS), axis=0
    )


def immersed_boundary_force(
    velocity_field: tuple[GridVariable, GridVariable],
    particles: list[callable],
    dirac_delta_approx: callable,
    surface_fn: callable,
    t: float,
    dt: float,
) -> tuple[GridVariable, GridVariable]:
    """
    Compute x and y components force from a array of immersed objects. Each object is represented as a point-cloud,
    as returned by the callable `particle.shape`.
    The 2-d velocity field is given by a GridArrayVector `velocity_field`.  `geom_param` and `Grid_p` are parameters passed
    to `shape_fn` required to compute the point cloud of the object.

    TODO: the use of `shape_fn` could and should be generalized.

    Args:
      velocity_field: the velocity field, i.e. vx or vy component
      particles: the immersed particles, represented as an iterable of callables; each callable
        in the list has to have a signature `x,y = f(float: t)`, with `t` the time, and `x,y,` the
        2-d point cloud representing the particle/object at time `t`.
      dirac_delta_approx: Approximation to the delta function
      surface_fn: Callable which computes the surface-integral `sum_{i,j} data[i,j] delta(x[i]-xp]) delta(y[j]-yp)  dx  dy)`
        for velocity_field[0].data and velocity_field[1].data (x- and y-components of the velocities)
      dt: The time step.

    Returns:
      tuple[GridVariable]: The total force field, i.e. x- and y-components of the force acting on the fluid velocities vx and vy,
        originating from all immersed objects. Each force Fx and Fy is defined on the same grid as vx and vy, respectively,
    """
    forcex = jnp.zeros_like(velocity_field[0].data)
    forcey = jnp.zeros_like(velocity_field[1].data)

    # run over all particles; the final force is the sum of all individual forces per particle
    for particle in particles:
        per_object_forcex, per_object_forcey = immersed_boundary_force_per_particle(
            velocity_field, particle, dirac_delta_approx, surface_fn, t, dt
        )
        forcex += per_object_forcex
        forcey += per_object_forcey
    return (
        GridVariable(
            GridArray(
                forcex,
                velocity_field[0].offset,
                velocity_field[0].grid,
                velocity_field[0].width,
            ),
            velocity_field[0].bc,
        ),
        GridVariable(
            GridArray(
                forcey,
                velocity_field[1].offset,
                velocity_field[1].grid,
                velocity_field[1].width,
            ),
            velocity_field[1].bc,
        ),
    )
