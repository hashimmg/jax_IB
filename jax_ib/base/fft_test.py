import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map as shmap
import numpy as np

import jax_ib.base.fft as fft

@pytest.fixture
def axis_names():
  return ('i','j')

@pytest.mark.parametrize("mesh_shape", [(1,8), (2,4), (4,2), (8,1)])
@pytest.mark.parametrize("N", [128])
@pytest.mark.parametrize("axis", [0,1])
def test_fft(N, axis, mesh_shape, axis_names):
  x = jnp.array(np.random.rand(N, N))
  mesh = jax.make_mesh(mesh_shape, axis_names)
  sharding = jax.sharding.NamedSharding(mesh, P(*axis_names))
  y = jax.device_put(x, sharding)
  jpfft = jax.jit(shmap(fft.fft, mesh=mesh, in_specs = (P(*axis_names),None,None), out_specs = P(*axis_names)), static_argnums = (1,2))
  actual = np.array(jpfft(y, axis, axis_names[axis]))
  expected = np.array(jnp.fft.fft(x, axis=axis))
  np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("mesh_shape", [(1,8), (2,4), (4,2), (8,1)])
@pytest.mark.parametrize("N", [128])
def test_fft_2d(N, mesh_shape, axis_names):
  x = jnp.array(np.random.rand(N, N))
  mesh = jax.make_mesh(mesh_shape, axis_names)
  sharding = jax.sharding.NamedSharding(mesh, P(*axis_names))
  y = jax.device_put(x, sharding)
  jpfft = jax.jit(shmap(fft.fft_2d, mesh=mesh, in_specs = (P(*axis_names),None), out_specs = P(*axis_names)), static_argnums = (1,))
  actual = np.array(jpfft(y, axis_names))
  expected = np.array(jnp.fft.fftn(x))
  np.testing.assert_allclose(actual, expected)

