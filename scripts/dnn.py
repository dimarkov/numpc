import os
import pprint
from re import sub

import sys
from tkinter import N

from numpy import inner
sys.path.append("..")

from pypc.datasets import MNIST

import numpy as np
import jax.numpy as jnp
from jax import random, nn, jit

from flax import linen as fnn

import seaborn as sns
import matplotlib.pyplot as plt

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive, HMCECS, NUTS, MCMC
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal

numpyro.enable_validation(True)

class DenseNet(fnn.Module):
  """A simple dense neural network."""

  @fnn.compact
  def __call__(self, x):
    n, d_x = x.shape
    x = fnn.Dense(features=300)(x)
    x = fnn.relu(x)
    x = fnn.Dense(features=100)(x)
    x = fnn.relu(x)
    x = fnn.Dense(features=10)(x)
    return x

train_dataset = MNIST(
    train=True, scale=None, size=None, normalize=False
)
test_dataset = MNIST(
    train=False, scale=None, size=None, normalize=False
)

n_classes = len(train_dataset.classes)

n_training = train_dataset.data.shape[0]
n_testing = test_dataset.data.shape[0]

train_x = jnp.array(train_dataset.data.numpy().reshape(n_training, -1) / 255.)
train_y = nn.one_hot(train_dataset.targets.numpy(), n_classes)

test_x = jnp.array(test_dataset.data.numpy().reshape(n_testing, -1) / 255.)
test_y = test_dataset.targets.numpy()


def model(nnet, init_params, x, y=None, subsample_size=None):
    n, d_x = x.shape

    if y is not None:
        _, d_y = y.shape
        assert y.shape[0] == n
    else:
        d_y = n_classes

    params = {}
    for key1 in init_params:
        params[key1] = {}
        for key2 in init_params[key1]:
            shape = init_params[key1][key2].shape
            if key2 == 'kernel':
                gs = numpyro.sample('gs' + '_' + key1, dist.HalfCauchy(1.).expand([shape[-1]]))
                ls = numpyro.sample('ls' + '_' + key1, dist.HalfCauchy(1.).expand(shape))
                params[key1][key2] = numpyro.sample(key1 + '_' + key2, dist.Normal(0., 0.1 * gs * ls).expand(shape))
            else:
                params[key1][key2] = numpyro.sample(key1 + '_' + key2, dist.Normal(0., 10.).expand(shape))

    with numpyro.plate("N", n, subsample_size=subsample_size):
        batch_x = numpyro.subsample(x, event_dim=1)
        logits = nnet().apply({'params': params}, batch_x)
        if y is not None:
            batch_y = numpyro.subsample(y, event_dim=1)
            numpyro.sample(
                "obs", dist.Categorical(logits=logits), obs=batch_y.argmax(-1)
            )
        else:
            numpyro.sample(
                "obs", dist.Categorical(logits=1e3*logits)
            )


dnn = DenseNet()
init_params = dnn.init(random.PRNGKey(0), jnp.ones((1, train_x.shape[-1])) )['params']
with numpyro.handlers.seed(rng_seed=0):
    model(DenseNet, init_params, train_x, y=train_y, subsample_size=64)

svi_key, mcmc_key = random.split(random.PRNGKey(0))

guide = AutoNormal(model)

# find reference parameters for second order taylor expansion to estimate likelihood (taylor_proxy)
optimizer = numpyro.optim.Adam(step_size=1e-3)
svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO(num_particles=10))
svi_result = svi.run(svi_key, 50000, DenseNet, init_params, train_x, y=train_y, subsample_size=64)
params, losses = svi_result.params, svi_result.losses
plt.plot(losses[-1000:])
plt.show()

predict = Predictive(model, guide=guide, params=params, num_samples=1000)
samples = predict(mcmc_key, DenseNet, init_params, test_x)

acc = np.array(jnp.mean(samples['obs'] == test_y, -1))
plt.hist( acc, bins=20)
plt.show()