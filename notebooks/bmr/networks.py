import jax.numpy as jnp
from jax import random
from jax.nn import one_hot
from flax import linen as fnn

from typing import Any, Callable, Sequence

class DenseNet(fnn.Module):
  """A simple dense neural network."""

  features: Sequence[int]
  act: Callable

  @fnn.compact
  def __call__(self, x, **kwargs):
    x = fnn.Dense(features=self.features[0], name='dense0')(x)
    for i, f in enumerate(self.features[1:]):
        x = self.act(x)
        x = fnn.Dense(features=f, name='dense{}'.format(i+1))(x)
    
    return 