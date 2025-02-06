"""
Configuration file for globally enabling gradient checkpointing
"""


import jax

CHECKPOINT = True

def enable_gradient_checkpoint():
  """
  Globally enable jax's gradient checkpointing.
  All functions wrapped with `jax_ib.base.config.checkpoint` are checkpointed.
  """
  global CHECKPOINT
  CHECKPOINT=True

def disable_gradient_checkpoint():
  """
  Globally disable jax's gradient checkpointing.
  All functions wrapped with `jax_ib.base.config.checkpoint` are no longer checkpointed.
  """

  global CHECKPOINT
  CHECKPOINT = False

def checkpoint(fun, *args, **kwargs):
  """
  Simple wrapper around jax.checkpoint. Use this function instead of
  jax.checkpoint to globally enable or disable checkpointing of all
  wrapped functions.
  """
  if CHECKPOINT:
    return jax.checkpoint(fun, *args, **kwargs)
  return fun
