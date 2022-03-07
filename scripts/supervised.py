import tqdm
import numpy as np
import jax.numpy as jnp
from jax import random, nn, jit, vmap

from flax import linen as fnn
from flax.core.frozen_dict import FrozenDict

from collections import defaultdict

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.contrib.module import flax_module

from numpc.datasets import MNIST

numpyro.set_platform('cpu')
# numpyro.enable_x64()
# numpyro.enable_validation(True)

class DenseLayer(fnn.Module):
  """A simple dense layer of a neural network."""
  features: int

  @fnn.compact
  def __call__(self, x):
    x = fnn.Dense(features=self.features)(x)
    x = fnn.relu(x)
    return x

def network_layers(features, image_dim):
    nnets = []
    d = image_dim
    L = len(features)
    for i, f in enumerate(features):
        nnet = flax_module('layer{}'.format(L - i - 1), DenseLayer(f), input_shape=(1, d))
        d = f
        nnets.append(nnet)    
    return nnets


def model(features, images, labels=None, subsample_size=None, likelihood='normal', sigma=1.):
    n, d = images.shape

    nnets = network_layers(features, d)
    L = len(nnets)
    
    inp_par = jnp.zeros(d)

    with numpyro.plate('batch', n, subsample_size=subsample_size):
        if images is not None:
            x_batch = numpyro.subsample(images, event_dim=1)
        else:
            x_batch = None

        l = 0
        x_out = numpyro.sample('x_{}'.format(L-l), dist.Normal(inp_par, sigma).to_event(1), obs=x_batch)

        for l in range(1, L + 1):
            pred = nnets[l-1](x_out)
            if L - l == 0:
                if labels is not None:
                    y_batch = numpyro.subsample(labels, event_dim=1)
                else:
                    y_batch = None
                    if likelihood == 'normal':
                        numpyro.sample('x_{}'.format(L-l), dist.Normal(pred, sigma).to_event(1), obs=y_batch)
                    elif likelihood == 'categorical':
                        numpyro.sample('x_{}'.format(L-l), dist.Categorical(logits=pred), obs=y_batch.argmax(-1))
                    else:
                        raise ValueError(f"Invalid choice of likelihood function.")
            else:
                x_out = numpyro.sample('x_{}'.format(L-l), dist.Normal(pred, sigma).to_event(1))


def test_model(test_data, features, params, likelihood='normal'):
    pred = Predictive(model, params=params, num_samples=1)
    sample = pred(_rng_key, features, test_data['image'], sigma=1e-6, likelihood=likelihood)

    pred_label = sample['x_0'][0]

    return np.mean(pred_label.argmax(-1) == test_data['label'])

# load data
train_ds, test_ds = MNIST()

# reshape images for dense networks
train_ds['image'] = train_ds['image'].reshape(train_ds['image'].shape[0], -1)
test_ds['image'] = test_ds['image'].reshape(test_ds['image'].shape[0], -1)


rng_key = random.PRNGKey(0)
batch_size = 64
features = [300, 100, 10]
with numpyro.handlers.seed(rng_seed=1):
    model(features, train_ds['image'], labels=train_ds['label'], subsample_size=batch_size)

guide = AutoDelta(model)
optimizer = numpyro.optim.ClippedAdam(step_size=1e-2, clip_norm=50.)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))

epochs = 30
num_iters = 1000
init_state = None
with tqdm.trange(epochs) as t:
    for _ in t:
        rng_key, _rng_key = random.split(rng_key)
        svi_result = svi.run(
            _rng_key, 
            num_iters, 
            features,
            train_ds['image'],
            progress_bar=False,
            init_state=init_state,
            labels=train_ds['label'],
            subsample_size=batch_size,
            likelihood='normal'
        )
        init_state = svi_result.state

        acc = test_model(test_ds, features, svi_result.params, likelihood='normal')
        
        t.set_postfix_str(
            "model elbo: {:.4f} - model test acc: {:.4f}".format(jnp.mean(svi_result.losses), acc),
            refresh=False,
        )