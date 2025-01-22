# fast fourier transformations on distributed hardware
import jax
import jax.numpy as jnp
from enum import Enum

class FFTDir(Enum):
  FWD='FWD'
  BWD='BWD'


FWD = FFTDir('FWD')
BWD = FFTDir('BWD')
def _get_sign(direction):
  assert direction in (FWD, BWD)
  if direction == FWD:
    return -1.0
  elif direction == BWD:
    return 1.0

def _get_transform(direction):
  assert direction in (FWD, BWD)
  if direction == FWD:
    return jnp.fft.fft
  elif direction == BWD:
    return jnp.fft.ifft

def _get_fft_i(axis_name: str, direction: FFTDir)-> jax.Array:
    """
    Compute the FFT of a 2d-array `array` along axis `0` of the array.
    `array` should be a jax.Array sharded on a device mesh ('i','j')
    `axis_name` is the name of the first sharding axis, i.e. `'i'` for
    the above mesh

    Args:
      array: A 2d jax Array to be transforemd
      axis_name: The name of the mapped axis `0`.

    Returns:
      jax.Array: The transformed `array`.
    """
    sign = _get_sign(direction)
    transform = _get_transform(direction)
    def fft(array):
        i = jax.lax.axis_index(axis_name)
        M = jax.lax.psum(1, axis_name=axis_name)
        N = M * array.shape[0]
        row = jnp.expand_dims(jnp.exp(sign*2*jnp.pi*1j/N*i*(N//M) * jnp.arange(N//M)), axis=1)
        result = jnp.zeros_like(array)
        for j in range(M):
            column = jnp.expand_dims(jnp.exp(sign *2*jnp.pi*1j/N*((j + i) % M) * (N//M) * jnp.arange(N//M)), axis=1)
            phase = jnp.exp(sign*2*jnp.pi*1j/N * i * ((j + i) % M) * (N//M)**2)
            result += column *transform(row *array, n = N, axis=0)[:(N//M),:]* phase
            array  = jax.lax.ppermute(array, axis_name=axis_name, perm = [(k, (k-1)%M) for k in range(M)])
        return result
    return fft

def _fft1d_i(array: jax.Array, axis_name: str)-> jax.Array:
  return _get_fft_i(axis_name, FWD)(array)

def _ifft1d_i(array: jax.Array, axis_name: str)-> jax.Array:
  return _get_fft_i(axis_name, BWD)(array)

def _get_fft_j(axis_name: str, direction: FFTDir)-> jax.Array:
    """
    Compute the FFT of a 2d-array `array` along axis `0` of the array.
    `array` should be a jax.Array sharded on a device mesh ('i','j')
    `axis_name` is the name of the first sharding axis, i.e. `'i'` for
    the above mesh

    Args:
      array: A 2d jax Array to be transforemd
      axis_name: The name of the mapped axis `0`.

    Returns:
      jax.Array: The transformed `array`.
    """
    sign = _get_sign(direction)
    transform = _get_transform(direction)
    def fft(array):
      j = jax.lax.axis_index(axis_name)
      M = jax.lax.psum(1, axis_name=axis_name)
      N = M * array.shape[1]
      column = jnp.expand_dims(jnp.exp(sign*2*jnp.pi*1j/N*j*(N//M) * jnp.arange(N//M)), axis=0)
      result = jnp.zeros_like(array)
      for i in range(M):
        row = jnp.expand_dims(jnp.exp(sign*2*jnp.pi*1j/N*((j + i) % M) * (N//M) * jnp.arange(N//M)), axis=0)
        phase = jnp.exp(sign*2*jnp.pi*1j/N * j * ((j + i) % M) * (N//M)**2)
        result += row * transform(array*column, n = N, axis=1)[:, :(N//M)]* phase
        array  = jax.lax.ppermute(array, axis_name=axis_name, perm = [(k, (k-1)%M) for k in range(M)])
      return result
    return fft

def _fft1d_j(array: jax.Array, axis_name: str)-> jax.Array:
  return _get_fft_j(axis_name, FWD)(array)

def _ifft1d_j(array: jax.Array, axis_name: str)-> jax.Array:
  return _get_fft_j(axis_name, BWD)(array)

def fft(array: jax.Array, axis:int, axis_name:str):
    """
    Compute the 1d-FFT of a 2d-array `array` along axis `axis` with name `axis_name`
    `array` should be a jax.Array  sharded on a device mesh ('i','j')

    Args:
      array: A 2d jax.Array to be transforemd
      axis: The axis to be transformed. Can be 0 or 1
      axis_name: The name of the mapped axis to be transforemd.

    This function should be `shard_map`ed and/or `jax.jit`ted.
    E.g. for a 8-device mesh of shape (4,2) with axis-names ('i','j'):
    ```python

    pfft = jax.jit(shard_map(fft, mesh=jax.make_mesh((4, 2), ('i', 'j')), in_specs = (PartitionSpec('i','j'),None,None), out_specs = PartitionSpec('i','j')), static_argmnums = (1,2))
    transformed_1 = pfft(array, 0, 'i')
    transformed_2 = pfft(array, 1, 'j')
    ```

    Returns:
      jax.Array: The transformed `array`.
    """
    if axis == 0:
        return _fft1d_i(array, axis_name)
    elif axis == 1:
        return _fft1d_j(array, axis_name)
    else:
        raise ValueError(f"axis {axis} not supported")

def ifft(array: jax.Array, axis:int, axis_name:str):
    """
    Compute the inverse 1d-FFT of a 2d-array `array` along axis `axis` with name `axis_name`
    `array` should be a jax.Array  sharded on a device mesh ('i','j')

    Args:
      array: A 2d jax.Array to be transforemd
      axis: The axis to be transformed. Can be 0 or 1
      axis_name: The name of the mapped axis to be transforemd.

    This function should be `shard_map`ed and/or `jax.jit`ted.
    E.g. for a 8-device mesh of shape (4,2) with axis-names ('i','j'):
    ```python

    pfft = jax.jit(shard_map(ifft, mesh=jax.make_mesh((4, 2), ('i', 'j')), in_specs = (PartitionSpec('i','j'),None,None), out_specs = PartitionSpec('i','j')), static_argmnums = (1,2))
    transformed_1 = pfft(array, 0, 'i')
    transformed_2 = pfft(array, 1, 'j')
    ```

    Returns:
      jax.Array: The transformed `array`.
    """
    if axis == 0:
        return _ifft1d_i(array, axis_name)
    elif axis == 1:
        return _ifft1d_j(array, axis_name)
    else:
        raise ValueError(f"axis {axis} not supported")

def fft_2d(array, axis_names):
    """
    Compute the 2d-FFT of a 2d-array `array`.
    `array` should be a jax.Array sharded on a device mesh
    with sharded axes with names `axis_names`

    This function should be `shard_map`ed and/or `jax.jit`ted.
    E.g. for a 8-device mesh of shape (4,2) with axis-names ('i','j'):
    ```python

    pfft_2d = jax.jit(shard_map(fft_2d, mesh=jax.make_mesh((4, 2), ('i', 'j')), in_specs = (PartitionSpec('i','j'),None), out_specs = PartitionSpec('i','j')), static_argmnums = (1,))
    transformed = pfft_2d(array, ('i','j')
    ```

    Args:
      array: A 2d jax Array to be transforemd
      axis_names: The names of the mapped axis of the device mesh.

    Returns:
      jax.Array: The transformed `array`.
    """

    return _fft1d_i(_fft1d_j(array, axis_names[1]), axis_names[0])

def ifft_2d(array, axis_names):
    """
    Compute the inverse 2d-FFT of a 2d-array `array`.
    `array` should be a jax.Array sharded on a device mesh
    with sharded axes with names `axis_names`

    This function should be `shard_map`ed and/or `jax.jit`ted.
    E.g. for a 8-device mesh of shape (4,2) with axis-names ('i','j'):
    ```python

    pfft_2d = jax.jit(shard_map(ifft_2d, mesh=jax.make_mesh((4, 2), ('i', 'j')), in_specs = (PartitionSpec('i','j'),None), out_specs = PartitionSpec('i','j')), static_argmnums = (1,))
    transformed = pfft_2d(array, ('i','j')
    ```

    Args:
      array: A 2d jax Array to be transforemd
      axis_names: The names of the mapped axis of the device mesh.

    Returns:
      jax.Array: The transformed `array`.
    """

    return _ifft1d_i(_ifft1d_j(array, axis_names[1]), axis_names[0])
