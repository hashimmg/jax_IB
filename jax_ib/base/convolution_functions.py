import jax
import jax.numpy as jnp
import pdb
from jax_ib.base.grids import GridVariable

def gaussian(x: jax.Array,mu: jax.Array,sigma:jax.Array)->float:
    """
    A standard gaussian. Used to approximate delta functions.

    Args:
      x: positions
      mu: location of the mean
      sigma: standard deviation of the gaussian

    Returns:
      float: The value of the gaussian
    """
    return 1/(sigma*jnp.sqrt(2*jnp.pi))*jnp.exp(-0.5*((x-mu)/sigma)**2)



# TODO: the dirac delta function should be removed as input
@jax.tree_util.Partial(jax.vmap, in_axes=(None, 0,0, None))
def convolve(field:GridVariable,xp:float,yp:float, dirac_delta_approx: callable) -> jax.Array:
    """
    Computes the 2-d convolution of the data in `fields.data` with a
    discrete approximation of the delta function. I.e. for a grid (i,j)
    with grid-points (x[i], y[j]), a 2d function data[i,j], and two points xp and yp,
    it computes

    `sum_{i,j} data[i,j] delta(x[i]-xp) delta(y[j]-yp)  dx  dy)`

    The point xp, yp does not have to be a grid point.
    The delta function requires the following signature:
    dirac_delta_approx(xp, X, dx)
    dirac_delta_approx(yp, Y, dy)

    with X, Y = grid.mesh() a mesh of the 2d grid, and dx, dx the grid spacing.

    Args:
      field: GridVariable whose data `field.data` to convolve as described above
      xp: x-value of the space-point of the surface of the object
      yp: y-value of the space-point of the surface of the object
      dirac_delta_approx: callable approximating a dirac-delta function

    Returns:
      jax.Array: the result of the convolution
    """
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dx = grid.step[0]
    dy = grid.step[1]
    return jnp.sum(field.data*dirac_delta_approx(xp,X,dx)*dirac_delta_approx(yp,Y,dy)*dx*dy)


def surf_fn_deprecated(field,xp,yp,discrete_fn):
    """
    Deprecated; use `convolve` above """
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dx = grid.step[0]
    dy = grid.step[1]
    def calc_force(xp,yp):
        return jnp.sum(field.data*discrete_fn(xp,X,dx)*discrete_fn(yp,Y,dy)*dx*dy)

    def foo(tree_arg):
        xp,yp = tree_arg
        return calc_force(xp,yp)

    def foo_pmap(tree_arg):
        return jax.vmap(foo,in_axes=1)(tree_arg)

    divider=jax.device_count()
    n = len(xp)//divider
    mapped = []
    for i in range(divider):
        mapped.append([xp[i*n:(i+1)*n],yp[i*n:(i+1)*n]])
    arr = jnp.array(mapped)
    U_deltas = jax.pmap(foo_pmap)(jnp.array(mapped))
    return U_deltas.flatten()  
