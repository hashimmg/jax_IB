import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax_ib.base.grids import GridVariable, GridArrayVector, GridArray
from jax_ib.base.particle_class import Particle

def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)


def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
   # offset = field.offset
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
   # X,Y =grid.mesh(offset)
    return integrate_trapz(field.data,dxEUL,dyEUL)


def IBM_force_GENERAL_deprecated(field,Xi,particle_center,geom_param,Grid_p,shape_fn,dirac_delta_approx,surface_fn,dx_dt,domega_dt,rotation,dt):
    """
    Deprecated: use immersed_boundary_force_per_particle
    Args:
      field: the velocity field, i.e. vx or vy component
      Xi: the axis of the component of the velocity, i.e. 0 if field = vx, and 1 if field = vy.
      particle_center: Xi-coordinate of the center-of-mass of the particle
      geom_param: The geometry parameters to compute the point-cloud of the shape of the object
      Grid_p: not required for flapping
      shape_fn: Function which returns the point-cloud of the object.
      dirac_delta_approx: approximation to the delta function
      surface_fn: Callable which computes the surface-integral `sum_{i,j} data[i,j] delta(x[i]-xp]) delta(y[j]-yp)  dx  dy)`
        for field.data
      dx_dt: callable to compute the speed along axis `Xi` of the geometric center of the object
      domega_dt: callable to compute the angular speed of rotation of the object
      rotation: callable which returns the current angle of the rotation for time t
    """
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    current_t = field.bc.time_stamp
    xp0,yp0 = shape_fn(geom_param,Grid_p)
    xp = xp0*jnp.cos(rotation(current_t))-yp0*jnp.sin(rotation(current_t))+particle_center[0]
    yp = xp0*jnp.sin(rotation(current_t))+yp0*jnp.cos(rotation(current_t))+particle_center[1]
    velocity_at_surface = surface_fn(field,xp,yp)

    if Xi==0: # mganahl: does Xi = 0 correspond to vx or vy?
        position_r = -(yp-particle_center[1])
    elif Xi==1:
        position_r = (xp-particle_center[0])

    U0 = dx_dt(current_t) # 2-d speed of the center of mass
    Omega=domega_dt(current_t)
    UP= U0[Xi] + Omega*position_r
    force = (UP - velocity_at_surface)/dt
   # if Xi==0:
        #plt.plot(xp,force)
        #maxforce =  delta_approx_logistjax(xp[0],X,0.003,1)
   #     maxforce = dirac_delta_approx(xp[3],X)
   #     plt.imshow(maxforce)
   #     print('Maxforce',jnp.max(maxforce))
    #    print(xp)
    x_i = jnp.roll(xp,-1)
    y_i = jnp.roll(yp,-1)
    dxL = x_i-xp
    dyL = y_i-yp
    dS = jnp.sqrt(dxL**2 + dyL**2)

    def calc_force(F,xp,yp,dss):
        # mganahl: Need to understand this better. In the code the dirac_delta_approx was formerly called discrete_fn, and they used a gaussian for that.
        #          All I did was renaming the function. Why computing the squares here again?
        return F*dirac_delta_approx(jnp.sqrt((xp-X)**2 + (yp-Y)**2),0,dxEUL)*dss
        #return F*dirac_delta_approx(xp-X,0,dxEUL)*dirac_delta_approx(yp-Y,0,dyEUL)*dss
        #return F*dirac_delta_approx(xp,X,dxEUL)*dirac_delta_approx(yp,Y,dyEUL)*dss**2
    def foo(tree_arg):
        F,xp,yp,dss = tree_arg
        print(F.shape, xp.shape, yp.shape, dss.shape)
        return calc_force(F,xp,yp,dss)

    def foo_pmap(tree_arg):
        print(tree_arg.shape)
        o = jax.vmap(foo,in_axes=1)(tree_arg)
        print('o', o.shape)
        bla = jnp.sum(jax.vmap(foo,in_axes=1)(tree_arg),axis=0)
        print('bla',bla.shape)
        return bla

    divider=jax.device_count()
    n = len(xp)//divider
    mapped = []
    for i in range(divider):
       mapped.append([force[i*n:(i+1)*n],xp[i*n:(i+1)*n],yp[i*n:(i+1)*n], dS[i*n:(i+1)*n]])

    #mapped = jnp.array([force,xp,yp])
    #remapped = mapped.reshape(())#jnp.array([[force[:n],xp[:n],yp[:n]],[force[n:],xp[n:],yp[n:]]])

    #return cfd.GridArray(jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0),offset,grid)
    out = jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0)
    #out = jnp.sum(foo_pmap(force, xp, yp, dS),axis=0)
    #out = foo_pmap(force, xp, yp, dS)
    print(out.shape)
    return out


def immersed_boundary_force_per_particle(
    velocity_field: tuple[GridVariable, GridVariable],
    particle_center: jax.Array,
    geom_param:jax.Array,
    Grid_p:jax.Array,
    shape_fn: callable,
    dirac_delta_approx: callable,
    surface_fn: callable,
    dx_dt: callable,
    dalpha_dt: callable,
    rotation: callable,
    dt:float)->jax.Array:
    """
    Compute the x and y forces from an immersed object. The object is represented as a point-cloud, as returned by `shape_fn`.
    The 2-d velocity field is given by a GridArrayVector `velocity_field`.  `geom_param` and `Grid_p` are parameters passed
    to `shape_fn` required to compute the point cloud of the object.

    TODO: the use of `shape_fn` could and should be generalized.

    Args:
      velocity_field: the velocity field, i.e. vx or vy component
      particle_center: The center-of-mass coordinates of the particle at time t.
      geom_param: The geometry parameters to compute the point-cloud of the shape of the object
      Grid_p: Not required for flapping
      shape_fn: Function which returns the point-cloud of the object.
      dirac_delta_approx: Approximation to the delta function
      surface_fn: Callable which computes the surface-integral `sum_{i,j} data[i,j] delta(x[i]-xp]) delta(y[j]-yp)  dx  dy)`
        for field.data
      dx_dt: Callable to compute the speed of geometric center of the object
      dalpha_dt: Callable to compute the angular speed of rotation of the object
      rotation: Callable which returns the current angle of the rotation of the object at time t. Required to compute the
        state geometric placement of the object at time `t`.
      dt: The time step.

    Returns:
      jax.Array of shape grid.shape: The forces Fx and Fy acting on the velocity fields vx and vy due to the presence of the object
        shape_fn (represented as a point cloud)
    """
    ux, uy = velocity_field
    grid = ux.grid
    offset = ux.offset
    X,Y = grid.mesh(offset) #2d is hard coded right now
    dx = grid.step[0]
    #dy = grid.step[1] # uncomment for different cal_force functions below
    current_t = ux.bc.time_stamp
    xp0,yp0 = shape_fn(geom_param,Grid_p)
    xp = xp0*jnp.cos(rotation(current_t))-yp0*jnp.sin(rotation(current_t))+particle_center[0]
    yp = xp0*jnp.sin(rotation(current_t))+yp0*jnp.cos(rotation(current_t))+particle_center[1]
    ux_at_surface = surface_fn(ux,xp,yp)
    uy_at_surface = surface_fn(uy,xp,yp)

    U0 = dx_dt(current_t) # 2-d speed of the center of mass
    Omega=dalpha_dt(current_t) #
    # include angular rotation of the object
    UPx= U0[0] - Omega*(yp-particle_center[1])
    UPy= U0[1] + Omega*(xp-particle_center[0])
    forcex = (UPx - ux_at_surface)/dt
    forcey = (UPy - uy_at_surface)/dt

    x_i = jnp.roll(xp,-1)
    y_i = jnp.roll(yp,-1)
    dxL = x_i-xp
    dyL = y_i-yp
    dS = jnp.sqrt(dxL**2 + dyL**2)

    def calc_force(F,xp,yp,dss):
        return F*dirac_delta_approx(jnp.sqrt((xp-X)**2 + (yp-Y)**2),0,dx)*dss
        #return F*dirac_delta_approx(xp-X,0,dx)*dirac_delta_approx(yp-Y,0,dy)*dss
        #return F*dirac_delta_approx(xp,X,dx)*dirac_delta_approx(yp,Y,dy)*dss**2

    vmapped_calc_force = jax.vmap(calc_force, in_axes=0)
    return jnp.sum(vmapped_calc_force(forcex, xp, yp, dS), axis=0), jnp.sum(vmapped_calc_force(forcey, xp, yp, dS), axis=0)



def immersed_boundary_force(velocity_field: tuple[GridVariable, GridVariable],
                            particles: Particle,
                            dirac_delta_approx: callable,
                            surface_fn: callable,
                            dt: float) -> tuple[GridVariable, GridVariable]:
    """
    Compute x and y components force from a array of immersed objects. Each object is represented as a point-cloud, 
    as returned by the callable `particle.shape`.
    The 2-d velocity field is given by a GridArrayVector `velocity_field`.  `geom_param` and `Grid_p` are parameters passed
    to `shape_fn` required to compute the point cloud of the object.

    TODO: the use of `shape_fn` could and should be generalized.

    Args:
      velocity_field: the velocity field, i.e. vx or vy component
      particles: Container class for assembling information about the immersed objects
      dirac_delta_approx: Approximation to the delta function
      surface_fn: Callable which computes the surface-integral `sum_{i,j} data[i,j] delta(x[i]-xp]) delta(y[j]-yp)  dx  dy)`
        for velocity_field[0].data and velocity_field[1].data (x- and y-components of the velocities)
      dt: The time step.

    Returns:
      tuple[GridVariable]: The total force field, i.e. x- and y-components of the force acting on the fluid velocities vx and vy,
        originating from all immersed objects. Each force Fx and Fy is defined on the same grid as vx and vy, respectively,


    """
    Grid_p = particles.generate_grid() #not required for flapping demo
    shape_fn = particles.shape
    Displacement_EQ = particles.Displacement_EQ
    Rotation_EQ = particles.Rotation_EQ
    Nparticles = len(particles.particle_center)
    particle_center = particles.particle_center
    geom_param = particles.geometry_param
    displacement_param = particles.displacement_param
    rotation_param = particles.rotation_param
    forcex = jnp.zeros_like(velocity_field[0].data)
    forcey = jnp.zeros_like(velocity_field[1].data)
    # run over all particles; the final force is the sum of all individual forces per particle
    for i in range(Nparticles):
        Xc = lambda t:Displacement_EQ([displacement_param[i]],t)
        rotation = lambda t:Rotation_EQ([rotation_param[i]],t)
        dx_dt = jax.jacrev(Xc) # return a vector of x- and y-speeds of center of mass
        domega_dt = jax.jacrev(rotation)
        per_object_forcex, per_object_forcey = immersed_boundary_force_per_particle(
          velocity_field, particle_center[i],geom_param[i],Grid_p,shape_fn,dirac_delta_approx,
          surface_fn,dx_dt,domega_dt,rotation,dt)
        forcex += per_object_forcex
        forcey += per_object_forcey
    return (GridVariable(GridArray(forcex,velocity_field[0].offset,velocity_field[0].grid), velocity_field[0].bc),
            GridVariable(GridArray(forcey,velocity_field[1].offset,velocity_field[1].grid), velocity_field[1].bc))

