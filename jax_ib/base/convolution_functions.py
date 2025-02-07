import jax
import jax.numpy as jnp
import pdb
from jax_ib.base.grids import GridVariable


def gaussian(x: jax.Array, mu: jax.Array, sigma: jax.Array) -> float:
    """
    A standard gaussian. Used to approximate delta functions.

    Args:
      x: positions
      mu: location of the mean
      sigma: standard deviation of the gaussian

    Returns:
      float: The value of the gaussian
    """
    return 1 / (sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


def mesh_convolve(
    field: GridVariable,
    x: jax.Array,
    dirac_delta_approx: callable,
    axis_names: list[str],
    vmapped: bool = False,
) -> jax.Array:
    """
    Compute the convolution of sharded array `field` with 2d-dirac-delta functions located at `x, y`.

    Args:
      field: GridVariable of the field
      x: 2-d locations of the dirac-delta peaks, shape (2, N)
      dirac_delta_approx: Function approximating a dirac-delta function in 1d.
        Expected function signature is `dirac_delta_approx(x, X, dx)`, with
        `x` a float, `X` a `jax.Array` of shape `field.data.shape`, and `dx`
        a float.
      axis_names: The names of the mapped axes of the device mesh.

    Returns:
      jax.Array: the convolution result.
    """
    if vmapped:
        conv = jax.vmap(convolve, in_axes=(None, 0, None))
    else:
        conv = _sequential_conv

    local_conv = conv(field, x, dirac_delta_approx)
    return jax.lax.psum(
        jax.lax.psum(local_conv, axis_name=axis_names[0]), axis_name=axis_names[1]
    )


def _sequential_conv(
    field: GridVariable, x: jax.Array, dirac_delta_approx: callable
) -> jax.Array:

    def body(carry, _x):
        y = jnp.sum(convolve(carry, _x, dirac_delta_approx))
        return carry, y

    _, result = jax.lax.scan(f=body, init=field, xs=x)
    return result


# TODO: the dirac delta function should be removed as input
def convolve(
    field: GridVariable, x: jax.Array, dirac_delta_approx: callable
) -> jax.Array:
    """
    Computes the 2-d convolution of the data in `fields.data` with a
    discrete approximation of the delta function. I.e. for a grid (i,j)
    with grid-points (_x[i], _y[j]), a 2d function data[i,j], and a 2d point points `x`
    it computes

    `sum_{i,j} data[i,j] delta(_x[i]-x[0]) delta(_y[j]-x[1])  dx  dy)`

    The point `x` does not have to be a grid point.
    The delta function requires the following signature:
    dirac_delta_approx(x[0], X, dx)
    dirac_delta_approx(x[1], Y, dy)

    with X, Y = grid.mesh() a mesh of the 2d grid, and dx, dx the grid spacing.

    Args:
      field: GridVariable whose data `field.data` to convolve as described above
      x: (x,y)-values of the space-points of the surface of the object
      dirac_delta_approx: callable approximating a dirac-delta function

    Returns:
      jax.Array: the result of the convolution
    """
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    dx = grid.step[0]
    dy = grid.step[1]
    return jnp.sum(
        field.data
        * dirac_delta_approx(x[0], X, dx)
        * dirac_delta_approx(x[1], Y, dy)
        * dx
        * dy
    )
