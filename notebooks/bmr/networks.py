import math
from typing import Optional, Callable, Sequence, List

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
from jaxtyping import Array

from equinox.nn import Linear
from equinox import Module, static_field

from flax import linen as fnn

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

def _identity(x):
    return x

class MLP(Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    layers: List[Linear]
    activation: Callable
    final_activation: Callable
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = ()

        bias = int(use_bias)

        if depth == 0:
            layers += (Linear(in_size + bias, out_size, use_bias=False, key=keys[0]), )
        else:
            layers += (Linear(in_size + bias, width_size, use_bias=False, key=keys[0]), )
            for i in range(depth - 1):
                layers += (Linear(width_size + bias, width_size, use_bias=False, key=keys[i + 1]), )
            layers += (Linear(width_size + bias, out_size, use_bias=False, key=keys[-1]), )
        
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.use_bias = use_bias

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            if self.use_bias:
                x = jnp.pad(x, (0, 1), constant_values=1.)
            x = layer(x)
            x = self.activation(x)

        if self.use_bias:
            x = jnp.pad(x, (0, 1), constant_values=1.)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x