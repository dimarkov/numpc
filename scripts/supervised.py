# set environment for better memory menagment on gpu
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
######################################################

from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from jax import random, nn, jit, vmap

from flax import linen as fnn
from flax.core.frozen_dict import FrozenDict

from collections import defaultdict

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from numpc.datasets import MNIST

numpyro.set_platform('cpu')
# numpyro.enable_validation(True)

class DenseLayer(fnn.Module):
  """A simple dense layer of a neural network."""
  features: int

  @fnn.compact
  def __call__(self, x):
    x = fnn.Dense(features=self.features)(x)
    x = fnn.relu(x)
    return x

def states(nnets, params, images, labels=None, subsample_size=None):
    n, _ = images.shape
    L = len(params) - 1

    with numpyro.plate('batch', n, subsample_size=subsample_size):
        if images is not None:
            x_batch = numpyro.subsample(images, event_dim=1)
        else:
            x_batch = None

        l = 0
        x_out = numpyro.sample('x_{}'.format(L-l), dist.Normal(params[l]['input'], 1.).to_event(1), obs=x_batch)

        for l in range(1, L):
            pred = nnets[l-1].apply(params[l]['loc'], x_out)
            if l == 0:
                if labels is not None:
                    y_batch = numpyro.subsample(labels, event_dim=1)
                else:
                    y_batch = None
                    numpyro.sample('x_{}'.format(L-l), dist.Normal(pred, 1.).to_event(1), obs=y_batch)
            else:
                x_out = numpyro.sample('x_{}'.format(L-l), dist.Normal(pred, 1.).to_event(1))

def weights(mus, params, nnets):
    L = len(mus) - 1
    for i, p in enumerate(params):
        if 'input' in p:
            pass
        else:
            s = {}
            for key in p['loc']['params']:
                s[key] = {}
                # loc_W = p[key]['kernel']
                # loc_b = p[key]['bias']

                scale_W = p['scale'][key]['kernel']
                scale_b = p['scale'][key]['bias']

                s[key]['kernel'] = numpyro.sample('W_{}-'.format(L - i) + key, dist.Normal(0., scale=scale_W).to_event(2))
                s[key]['bias'] = numpyro.sample('b_{}-'.format(L - i) + key, dist.Normal(0., scale=scale_b).to_event(1))

            pred = nnets[i-1].apply( {'params': s}, mus['x_{}'.format(L - i + 1)] )
            with numpyro.plate('batch_{}'.format(i), pred.shape[0]):
                numpyro.sample('mu_{}'.format(L - i), dist.Normal(pred, 1.).to_event(1), obs=mus['x_{}'.format(L - i)])

def infer_states(rng_key, data, params, nnets, subsample_size=64, num_iters=100):
    optimizer = numpyro.optim.SGD(step_size=1e-2)
    guide = AutoDelta(states)
    svi = SVI(states, guide, optimizer, loss=Trace_ELBO(num_particles=1))
    svi_result = svi.run(rng_key, num_iters, nnets, params, data['image'], data['label'], subsample_size=subsample_size, progress_bar=False)

    mus = {'x_3': data['image']}
    for i in range(n_layers-1, 0, -1):
        mus['x_{}'.format(i)] = svi_result.params['x_{}_auto_loc'.format(i)]
    
    mus['x_0'] = nn.one_hot(data['label'], 10)

    return mus

def format_parameters(params):
    locs = defaultdict(lambda: defaultdict(lambda: {}))
    # scl = {}
    for (key, value) in params.items():
        s1, s2 = key.split('-')
        k = s2.split('_')
        layer_name = k[0] + '_' + k[1]
        label, layer = s1.split('_')

        # scl[s2] = {'kernel': None, 'bias': None}
        if label == 'W':
            locs[layer][layer_name]['kernel'] = value
            # scl[k]['kernel'] = value.std(0, ddof=1)
        elif label == 'b':
            locs[layer][layer_name]['bias'] = value
            # scl[k]['bias'] = value.std(0, ddof=1)

    return locs

def learn_parameters(rng_key, mus, params, nnets, num_iters=100):
    optimizer = numpyro.optim.Adam(step_size=1e-4)
    guide = AutoDelta(weights)
    svi = SVI(weights, guide, optimizer, loss=Trace_ELBO(num_particles=1))
    svi_result = svi.run(rng_key, num_iters, mus, params, nnets, progress_bar=False)

    return format_parameters(svi_result.params)

def test_model(rng_key, test_data, params, nnets):
    x = test_data['image']
    for i, nnet in enumerate(nnets):
        x = nnet.apply(params[i + 1]['loc'], x)

    print('model acc :', np.mean(x.argmax(-1) == test_data['label']))

# load data
train_ds, test_ds = MNIST()

# reshape images for dense networks
train_ds['image'] = train_ds['image'].reshape(train_ds['image'].shape[0], -1)
test_ds['image'] = test_ds['image'].reshape(test_ds['image'].shape[0], -1)

####################### initialise network parameters #############
rng_key = random.PRNGKey(0)
x_init = train_ds['image'][0]
features = [300, 100, 10]
n_layers = len(features)

nnets = []
params= [{'input': jnp.zeros(x_init.shape[-1])}]
for l, f in enumerate(features):
    nnet = DenseLayer(features=f)
    rng_key, _rng_key = random.split(rng_key)
    init_params = nnet.init(_rng_key, x_init)
    x_init = nnet.apply(init_params, x_init)
    nnets.append(nnet)
    params.append({'loc': init_params})
    p = init_params['params']
    scl = {}
    for key in p:
        scl[key] = {
                'kernel': 1e1 * jnp.ones_like(p[key]['kernel']), 
                'bias': 1e1 * jnp.ones_like(p[key]['bias']) 
        }
    params[-1]['scale'] = scl
####################################################################

for i in tqdm(range(30)):
  
    rng_key, _rng_key = random.split(rng_key)
    mus = infer_states(_rng_key, train_ds, params, nnets, num_iters=1000, subsample_size=64)

    rng_key, _rng_key = random.split(rng_key)
    results = learn_parameters(_rng_key, mus, params, nnets, num_iters=100)

    for s in results:
        i = int(s)
        params[-1 - i]['loc'] = FrozenDict({'params': results[s]})
        
    rng_key, _rng_key = random.split(rng_key)
    test_model(_rng_key, test_ds, params, nnets)