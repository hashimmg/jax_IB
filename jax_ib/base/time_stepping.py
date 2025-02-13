import dataclasses
from functools import partial
from typing import Callable, Sequence, TypeVar, Optional
import jax
import jax.numpy as jnp
import tree_math
from jax_ib.base import boundaries
from jax_ib.base.particle_class import All_Variables
from jax_ib.base import (
    grids,
    fast_diagonalization,
    finite_differences,
    IBM_Force,
    convolution_functions,
    pressure as prs,
)
from jax_ib.base import time_stepping, config, particle_class
from jax_ib.base.config import checkpoint


PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]
from jax_ib.base.grids import GridVariable, GridArray


class ExplicitNavierStokesODE_Penalty:
    """Spatially discretized version of Navier-Stokes.

    The equation is given by:

      ∂u/∂t = explicit_terms(u)
      0 = incompressibility_constraint(u)
    """

    def __init__(self, explicit_terms, pressure_projection, update_BC, Reserve_BC):
        self.explicit_terms = explicit_terms
        self.pressure_projection = pressure_projection
        self.update_BC = update_BC
        self.Reserve_BC = Reserve_BC

    def explicit_terms(self, state):
        """Explicitly evaluate the ODE."""
        raise NotImplementedError

    def pressure_projection(self, state):
        """Enforce the incompressibility constraint."""
        raise NotImplementedError

    def update_BC(self, state):
        """Update Wall BC"""
        raise NotImplementedError

    def Reserve_BC(self, state):
        """Revert spurious updates of Wall BC"""
        raise NotImplementedError


class ExplicitNavierStokesODE_BCtime:
    """Spatially discretized version of Navier-Stokes.

    The equation is given by:

      ∂u/∂t = explicit_terms(u)
      0 = incompressibility_constraint(u)
    """

    def __init__(
        self,
        explicit_terms,
        pressure_projection,
        update_BC,
        Reserve_BC,
        IBM_force,
        Pressure_Grad,
        Calculate_Drag,
    ):
        self.explicit_terms = explicit_terms
        self.pressure_projection = pressure_projection
        self.update_BC = update_BC
        self.Reserve_BC = Reserve_BC
        self.IBM_force = IBM_force
        self.Pressure_Grad = Pressure_Grad
        self.Calculate_Drag = Calculate_Drag

    def explicit_terms(self, state):
        """Explicitly evaluate the ODE."""
        raise NotImplementedError

    def pressure_projection(self, state):
        """Enforce the incompressibility constraint."""
        raise NotImplementedError

    def update_BC(self, state):
        """Update Wall BC"""
        raise NotImplementedError

    def Reserve_BC(self, state):
        """Revert spurious updates of Wall BC"""
        raise NotImplementedError

    def IBM_force(self, state):
        """Revert spurious updates of Wall BC"""
        raise NotImplementedError

    def Pressure_Grad(self, state):
        """Revert spurious updates of Wall BC"""
        raise NotImplementedError

    def Calculate_Drag(self, state):
        """Revert spurious updates of Wall BC"""
        raise NotImplementedError


@dataclasses.dataclass
class ButcherTableau_updated:
    a: Sequence[Sequence[float]]
    b: Sequence[float]
    c: Sequence[float]
    # TODO(shoyer): add c, when we support time-dependent equations.

    def __post_init__(self):
        if len(self.a) + 1 != len(self.b):
            raise ValueError("inconsistent Butcher tableau")


def navier_stokes_rk_updated(
    tableau: ButcherTableau_updated,
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
    """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.

    This function implements the reference method (equations 16-21), rather than
    the fast projection method, from:
      "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
      Fluids 2020, 5, 222; doi:10.3390/fluids5040222

    Args:
      tableau: Butcher tableau.
      equation: equation to use.
      time_step: overall time-step size.

    Returns:
      Function that advances one time-step forward.
    """
    # pylint: disable=invalid-name
    dt = time_step
    explicit_terms = tree_math.unwrap(
        equation.explicit_terms
    )  # explicit update function, takes velocity (tuple[GridVariable]) as input
    pressure_projection = tree_math.unwrap(
        equation.pressure_projection
    )  # takes velocity and pressure and returns updated velocity and pressure

    IBM = equation.IBM_force
    Grad_Pressure = tree_math.unwrap(equation.Pressure_Grad)

    a = tableau.a
    b = tableau.b
    num_steps = len(b)
    update_BC = equation.update_BC

    def step_fn(all_variables: All_Variables):
        u = [None] * num_steps
        k = [None] * num_steps
        time = all_variables.time

        def pressure_gradient_to_GridVariable(pressure_gradient, bcs):
            """
            unwrap pressure_gradient (tm.Vector[tuple[GridArray, GridArray]]
            and rewrap into tm.Vector[tuple[GridVariable, GridVariable]
            """
            return tree_math.Vector(
                tuple(
                    grids.GridVariable(dp, bc)
                    for dp, bc in zip(pressure_gradient.tree, bcs)
                )
            )

        ubc = tuple([v.bc for v in all_variables.velocity])
        pressure = tree_math.Vector(all_variables.pressure)

        velocity_field = tree_math.Vector(all_variables.velocity)
        u[0] = velocity_field
        k[0] = explicit_terms(velocity_field)

        dP = pressure_gradient_to_GridVariable(Grad_Pressure(pressure), ubc)
        u0 = velocity_field
        for i in range(1, num_steps):
            u_star = u0 + dt * sum(a[i - 1][j] * k[j] for j in range(i) if a[i - 1][j])
            u_temp, _ = pressure_projection(pressure, u_star).tree
            u[i] = tree_math.Vector(u_temp)
            k[i] = explicit_terms(u[i])

        # mganahl: why is dP below not multiplied by dt?
        u_star = (
            u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j]) - dP * dt
        )  # mganahl clarify with Mohammed correctness

        Force = tree_math.Vector(IBM(u_star.tree, time, dt))
        u_star_star = u_star + dt * Force

        u_final, new_pressure = pressure_projection(pressure, u_star_star).tree
        updated_variables = All_Variables(
            u_final,
            new_pressure,
            all_variables.Drag,
            all_variables.Step_count + 1,
            all_variables.MD_var,
            time + dt,
        )
        return updated_variables

    return step_fn


def get_step_fn_sharded(
    laplacian_eigenvalues: tuple[jax.Array, jax.Array],
    dt: float,
    width: int,
    obj_fn: callable,
    explicit_update_fn: callable,
    axis_names: tuple[str],
    enable_checkpointing: bool = True,
) -> callable:
    """
    Return the function that performes an update step of pressure and velocity
    fields for the incompressible Navier Stokes equation.

    Args:
      laplacian_eigenvalues: The eigenvalues of the iscretized 1d laplacian operators
        for each spatial dimension
      dt: The time step
      width: The padding width for the local shards of the arrays
      obj_fn: Callable returning immersed objects as a point cloud at asolute time `t`.
        Signature is `x, y = obj_fn(t)`. The function has to be jax differentiable w.r.t. `t`.
      explicit_update_fn: A callable with signature
        `explicit_update_fn(tuple[GridVariable, GridVariable]) -> tuple[GridVariable, GridVariable]`.
        Computes the local update of velocities coresponding to advection, diffusion and force-terms.
      axis_names: The names of the mapped axes
      enable_checkpinting: If True, the step function is checkpointed at every step

    Returns:
      callable: The step function for a single update step. The signature of the this function is
        step_fn(args: tuple[pressure: GridVariable, velocities: tuple[GridVariable, GridVariable], time: float]) -> tuple[GridVariable, tuple[GridVariable, GridVariable], float]
    """
    cutoff = 10 * jnp.finfo(jnp.float32).eps
    eigvals = jnp.add.outer(laplacian_eigenvalues[0], laplacian_eigenvalues[1].T)
    pinv = fast_diagonalization.pseudo_poisson_inversion(
        eigvals, jnp.complex128, ("i", "j"), cutoff
    )

    surface_velocity_fn = lambda f, x: convolution_functions.mesh_convolve(
        f, x, convolution_functions.gaussian, axis_names=axis_names, vmapped=False
    )

    @partial(checkpoint, checkpointing=enable_checkpointing)
    def step_fn(
        args: tuple[GridVariable, tuple[GridVariable, GridVariable], float],
    ) -> tuple[GridVariable, tuple[GridVariable, GridVariable], float]:
        """single  update step"""
        p, us, _, t = args

        local_velocities = tuple([u.shard_pad(width) for u in us])
        temp = tuple([v.crop(width) for v in explicit_update_fn(local_velocities)])

        del local_velocities

        us[0].array.data += dt * temp[0].data
        us[1].array.data += dt * temp[1].data
        local_pressure = p.shard_pad(width)
        temp = tuple(
            [
                dp.crop(width)
                for dp in finite_differences.forward_difference(local_pressure)
            ]
        )
        del local_pressure

        # mganahl: clarify with Mohammed if this is correct
        # this deviates from the original implementation
        us[0].array.data -= dt * temp[0].data
        us[1].array.data -= dt * temp[1].data

        force = IBM_Force.immersed_boundary_force(
            us, [obj_fn], convolution_functions.gaussian, surface_velocity_fn, t, dt
        )

        us[0].array.data += dt * force[0].data
        us[1].array.data += dt * force[1].data

        local_u_projected, local_pressure = prs.projection_and_update_pressure_sharded(
            p, us, pinv, width
        )
        local_pressure.array.grid = p.grid
        local_u_projected[0].array.grid = us[0].grid
        local_u_projected[1].array.grid = us[1].grid

        return local_pressure, local_u_projected, (force[0].data, force[1].data), t + dt

    return step_fn


def evolve_navier_stokes_sharded(
    pressure: GridVariable,
    velocities: tuple[GridVariable],
    laplacian_eigenvalues: tuple[jax.Array],
    reference_time: float,
    dt: float,
    width: int,
    inner_steps: int,
    outer_steps: int,
    obj_fn: callable,
    explicit_update_fn: callable,
    axis_names: tuple[str],
    data_processing_inner: Optional[callable] = None,
    data_processing_outer: Optional[callable] = None,
    stepfn_checkpointing: bool = False,
    inner_checkpointing: bool = False,
    outer_checkpointing: bool = True,
):
    """
    Evolve `pressure` and velocities` using, the incompressible Navier Stokes equation.

    Args:
      pressure: Pressure variable on a grid.
      velocities: Tuple of velocity variables on a grid.
      laplacian_eigenvalues: The eigenvalues of the iscretized 1d laplacian operators
        for each spatial dimension
      reference_time: The initial time. Required to compute the values of `obj_fn(reference_time + n_steps * dt)`,
        see below.
      dt: The time step
      width: The padding width for the local shards of the arrays
      inner_steps: Number of "inner" steps to perform. No data is accumulated during these steps.
      outer_steps: Number of "outer" steps to perform. Data is accumulated during these steps using jax.lax.scan.
      obj_fn: Callable returning immersed objects as a point cloud at asolute time `t`.
        Signature is `x, y = obj_fn(t)`. The function has to be jax differentiable w.r.t. `t`.
      explicit_update_fn: A callable with signature
        `explicit_update_fn(tuple[GridVariable, GridVariable]) -> tuple[GridVariable, GridVariable]`.
        Computes the local update of velocities coresponding to advection, diffusion and force-terms.
      axis_names: the names of the pmappe axes
      data_processing_inner: A function with signature
        data_processing_int(tuple[GridVariable, tuple[GridVariable, GridVariable], tuple[GridVariable, GridVariable],float]) -> Any
        The outpout of this function is accumulated using jax.lax.scan over the inner loop and
        passed into `data_processing_outer` during the outer loop. For `None`, defaults to `lambda *args: None`.
        The input variables correspond to
          - pressure
          - velocities
          - IBM-force updates
          - elapsed times
        respectively.
        If this function is wrapped with shard_map, out_specs have to match the pytree structure of the return value
        of `evolve_navier_stokes_sharded`, which may depend on the pytree structure of the return values of
        `data_processing_inner`.
      data_processing_outer: A function with signature
        data_processing_outer(tuple[GridVariable, tuple[GridVariable, GridVariable], tuple[GridVariable, GridVariable],float], Any) -> Any
        The output of the `data_processing_inner` is passed in as the second argument to `data_processing_outer`.
        The outpout of this function is accumulated using jax.lax.scan over the outer loop and returned
        to the caller. For `None`, defaults to `lambda *args: None`.
        The input corresponds to
          (
            - pressure
            - velocities
            - IBM-force updates
            - elapsed times
          )
          - return values of `data_processing_inner`.
        If this function is wrapped with shard_map, out_specs have to match the pytree structure of the return value
        of `evolve_navier_stokes_sharded`, which may depend on the pytree structure of the return values of
        `data_processing_outer` and `data_processing_inner`.

    Returns:
      GridVariable: The final pressure
      tuple[GridVariable, GridVariable]: The final velocities
      Any: The accumulated (stacked) output of `data_processing_outer` over the outer loops.
        For `data_processing_outer=None` equals `None`. The pytree structure of this output
        may depend on the pytree structure of the return values of `data_processing_inner` and
        `data_processing_outer`.
    """

    if data_processing_inner is None:
        data_processing_inner = lambda *args: None

    if data_processing_outer is None:
        data_processing_outer = lambda *args: None

    step_fun = get_step_fn_sharded(
        laplacian_eigenvalues,
        dt,
        width,
        obj_fn,
        explicit_update_fn,
        axis_names,
        stepfn_checkpointing,
    )

    i = jax.lax.axis_index("i")
    j = jax.lax.axis_index("j")

    # map quantities to local grid
    local_pressure = pressure.to_subgrid((i, j))
    local_velocities = tuple([u.to_subgrid((i, j)) for u in velocities])

    def _step_fun(carry, x):
        result = step_fun(carry)
        return result, data_processing_inner(result)

    # inner loop running over `inner_steps`
    @partial(checkpoint, checkpointing=inner_checkpointing)
    def inner(args):
        carry, y = jax.lax.scan(_step_fun, args, xs=None, length=inner_steps)
        return carry, y

    # outer loop running over `outer_steps`
    @partial(checkpoint,checkpointing=outer_checkpointing)
    def outer(carry, x):
        result, y = inner(carry)
        return result, data_processing_outer(result, y)

    # initial values for for_loop
    ibm_forces = tuple([jnp.zeros_like(v.data) for v in local_velocities])
    init = (local_pressure, local_velocities, ibm_forces, reference_time)
    carry, y = jax.lax.scan(outer, init, xs=None, length=outer_steps)
    final_pressure, final_velocities, _, _ = carry

    final_pressure.array.grid = pressure.grid
    final_velocities[0].array.grid = velocities[0].grid
    final_velocities[1].array.grid = velocities[1].grid
    # return results
    return final_pressure, final_velocities, y


def navier_stokes_rk_penalty(
    tableau: ButcherTableau_updated,
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
    """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.

    This function implements the reference method (equations 16-21), rather than
    the fast projection method, from:
      "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
      Fluids 2020, 5, 222; doi:10.3390/fluids5040222

    Args:
      tableau: Butcher tableau.
      equation: equation to use.
      time_step: overall time-step size.

    Returns:
      Function that advances one time-step forward.
    """
    raise NotImplementedError("Currenty not implemented")
    # pylint: disable=invalid-name
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    P = tree_math.unwrap(equation.pressure_projection)
    M = tree_math.unwrap(equation.update_BC)
    R = tree_math.unwrap(equation.Reserve_BC)

    a = tableau.a
    b = tableau.b
    num_steps = len(b)

    @tree_math.wrap
    def step_fn(u0):
        u = [None] * num_steps
        k = [None] * num_steps

        def convert_to_velocity_vecot(u0):
            u = u0.tree
            return tree_math.Vector(tuple(u[i].array for i in range(len(u))))

        def convert_to_velocity_tree(m, bcs):
            return tree_math.Vector(
                tuple(grids.GridVariable(v, bc) for v, bc in zip(m.tree, bcs))
            )

        def convert_all_variabl_to_velocity_vecot(u0):
            u = u0.tree.velocity
            # return tree_math.Vector(tuple(grids.GridVariable(v.array,v.bc) for v in u))
            return tree_math.Vector(u)

        def covert_veloicty_to_All_variable_vecot(
            particles, m, pressure, Drag, Step_count, MD_var
        ):
            u = m.tree
            # return tree_math.Vector(particle_class.All_Variables(particles, tuple(grids.GridVariable(v.array,v.bc) for v in u),pressure))
            return tree_math.Vector(
                particle_class.All_Variables(
                    particles, u, pressure, Drag, Step_count, MD_var
                )
            )

        def velocity_bc(u0):
            u = u0.tree.velocity
            return tuple(u[i].bc for i in range(len(u)))

        def the_particles(u0):
            return u0.tree.particles

        def the_pressure(u0):
            return u0.tree.pressure

        def the_Drag(u0):
            return u0.tree.Drag

        particles = the_particles(u0)
        ubc = velocity_bc(u0)
        pressure = the_pressure(u0)
        Drag = the_Drag(u0)
        Step_count = u0.tree.Step_count
        MD_var = u0.tree.MD_var

        u0 = convert_all_variabl_to_velocity_vecot(u0)

        u[0] = convert_to_velocity_vecot(u0)
        k[0] = convert_to_velocity_vecot(F(u0))

        u0 = convert_to_velocity_vecot(u0)

        for i in range(1, num_steps):
            # u_star = u0[ww].array + sum(a[i-1][j]*k[j][ww].array for j in range(i) if a[i-1][j])

            u_star = u0 + dt * sum(a[i - 1][j] * k[j] for j in range(i) if a[i - 1][j])

            # u[i] = P(R(u_star))
            u[i] = convert_to_velocity_vecot(P(convert_to_velocity_tree(u_star, ubc)))
            k[i] = convert_to_velocity_vecot(F(convert_to_velocity_tree(u[i], ubc)))

        # for ww in range(0,len(u0)):
        u_star = u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])

        u_final = convert_to_velocity_tree(u_star, ubc)
        u_final = covert_veloicty_to_All_variable_vecot(
            particles, u_final, pressure, Drag, Step_count, MD_var
        )
        u_final = P(u_final)
        #
        u_final = M(u_final)

        return u_final

    return step_fn


def forward_euler_penalty(
    equation: ExplicitNavierStokesODE_Penalty,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        navier_stokes_rk_penalty(
            ButcherTableau_updated(a=[], b=[1], c=[0]), equation, time_step
        ),
        name="forward_euler",
    )


def forward_euler_updated(
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        navier_stokes_rk_updated(
            ButcherTableau_updated(a=[], b=[1], c=[0]), equation, time_step
        ),
        name="forward_euler",
    )
