"""Fast diagonalization method for inverting linear operators."""
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import functools

import jax_ib.base.fft as fft
import jax_ib.base.fast_diagonalization as fdiag

@pytest.fixture
def axis_names():
  return ('i','j')


@pytest.mark.parametrize("mesh_shape", [(1,8), (2,4), (4,2), (8,1)])
@pytest.mark.parametrize("N", [128])
def test_fft_inversion(N, mesh_shape, axis_names):
  cutoff = 10 * jnp.finfo(jnp.float32).eps
  def func(v):
    return jnp.where(abs(v) > cutoff, 1 / v, 0)

  mesh = jax.make_mesh(mesh_shape, axis_names)
  sharding = jax.sharding.NamedSharding(mesh, P(*axis_names))

  circulants= [np.random.rand(N) for n in range(2)]
  rhs = np.random.rand(N, N)

  eigenvalues = [np.fft.fft(c) for c in circulants]
  summed_eigenvalues = functools.reduce(np.add.outer,eigenvalues)
  diagonals = jnp.asarray(func(summed_eigenvalues))
  expected = np.fft.ifftn(diagonals * np.fft.fftn(rhs))


  @jax.jit
  @functools.partial(shard_map, mesh=mesh, in_specs = (P('i'),P('j'), P('i','j')), out_specs = P('i','j'))
  def invert(x, y, rhs):
    eigvals = jnp.add.outer(x, y.T)
    inv = fdiag.pseudo_poisson_inversion(eigvals, rhs.dtype, axis_names, cutoff)
    return inv(rhs)

  actual = np.array(invert(eigenvalues[0], eigenvalues[1], rhs))
  np.testing.assert_allclose(expected, actual)

