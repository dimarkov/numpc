import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
######################################################

from typing import Any, Callable, Sequence, Optional

from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from jax import random, nn, jit, vmap, tree_map

from flax import linen as fnn
from flax.core.frozen_dict import FrozenDict

from collections import defaultdict

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from numpc.datasets import MNIST

numpyro.set_platform('gpu')

class DenseNet(fnn.Module):
  """A simple dense neural network."""

  features: Sequence[int]

  @fnn.compact
  def __call__(self, x):
    x = fnn.Dense(features=self.features[0])(x)
    x = fnn.relu(x)
    x = fnn.Dense(features=self.features[1])(x)
    x = fnn.relu(x)
    x = fnn.Dense(features=self.features[2])(x)

    return fnn.relu(x)


def model(nnet, init_params, images, sigma=1., labels=None, subsample_size=None):
    n, _ = images.shape

    params = defaultdict(lambda: {})
    for key1 in init_params:
        for key2 in init_params[key1]:
            params[key1][key2] = numpyro.param(key1 + '-' + key2, init_params[key1][key2])

    numpyro.deterministic('params', params)

    with numpyro.plate("N", n, subsample_size=subsample_size):
        batch_x = numpyro.subsample(images, event_dim=1)
        pred = nnet.apply({'params': FrozenDict(params)}, batch_x)
        if labels is not None:
            batch_y = numpyro.subsample(labels, event_dim=1)
        else:
            batch_y = None
        
        numpyro.sample(
                "obs", dist.Normal(pred, sigma).to_event(1), obs=batch_y
            )

rng_key = random.PRNGKey(0)

# load data
train_ds, test_ds = MNIST()

# reshape images for dense networks
train_ds['image'] = train_ds['image'].reshape(train_ds['image'].shape[0], -1)
test_ds['image'] = test_ds['image'].reshape(test_ds['image'].shape[0], -1)

# create network
dnn = DenseNet(features=[300, 100, 10])
init_params = dnn.init(random.PRNGKey(0), train_ds['image'] )['params']

guide = AutoDelta(model)
optimizer = numpyro.optim.Adam(step_size=1e-4)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))
rng_key, _rng_key = random.split(rng_key)
svi_result = svi.run(
    _rng_key, 
    30000, 
    dnn, 
    init_params, 
    train_ds['image'], 
    labels=nn.one_hot(train_ds['label'], 10), 
    subsample_size=64)

params, losses = svi_result.params, svi_result.losses

pred = Predictive(model, params=params, num_samples=1)
sample = pred(_rng_key, dnn, init_params, test_ds['image'], sigma=1e-6)

acc = np.mean( sample['obs'][0].argmax(-1) == test_ds['label'] )
print('model acc :', acc)