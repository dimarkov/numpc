import argparse
import os

from attr import mutable
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
######################################################

from functools import partial
from typing import Any, Callable, Sequence, Optional, Tuple

import jax.numpy as jnp
from jax import random, nn, device_put, devices
from flax import linen as fnn

import optax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.contrib.module import flax_module

from numpc.datasets import MNIST

class DenseNet(fnn.Module):
  """A simple dense neural network."""

  features: Sequence[int]
  act: Callable

  @fnn.compact
  def __call__(self, x, **kwargs):
    x = fnn.Dense(features=self.features[0])(x)
    for f in self.features[1:]:
        x = self.act(x)
        x = fnn.Dense(features=f)(x)
    
    return x

class LeNet(fnn.Module):
    """A simple convolutional network.
    example from https://github.com/FluxML/model-zoo/tree/master/vision/conv_mnist
    """
       
    conv_features: Sequence[int]
    dense_features: Sequence[int]
    kernel_size: Tuple[int] = (5, 5) 
    window_shape: Tuple[int] = (2, 2) 
    strides: Tuple[int] = (2, 2) 

    @fnn.compact
    def __call__(self, x, **kwargs):
        for f in self.conv_features:
            x = fnn.Conv(features=f, kernel_size=self.kernel_size)(x)
            x = fnn.relu(x)
            x = fnn.avg_pool(x, window_shape=self.window_shape, strides=self.strides)
        
        x = x.reshape((x.shape[0], -1))
        x = fnn.Dense(features=self.dense_features[0])(x)
        for f in self.dense_features[1:]:
            x = fnn.relu(x)
            x = fnn.Dense(features=f)(x)
        return x

ModuleDef = Any
# flax_module does not work with batch_norm layers for now
class ResNetBlock(fnn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  kernel_size: Tuple[int, int] = (4, 4)
  strides: Tuple[int, int] = (1, 1)

  @fnn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, self.kernel_size, self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, self.kernel_size)(y)
    y = self.norm(scale_init=fnn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)

class ResNet(fnn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 16
  dtype: Any = jnp.float32
  act: Callable = fnn.relu
  conv: ModuleDef = fnn.Conv

  @fnn.compact
  def __call__(self, x, train):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(fnn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = self.act(x)
    x = fnn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = fnn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x

def likelihood(nnet, images, labels, sigma, n, subsample_size, ll_type, event_dim=1, train=False):
    with numpyro.plate("N", n, subsample_size=subsample_size):
        batch_x = numpyro.subsample(images, event_dim=event_dim)
        pred = nnet(batch_x, train=train)
        
        if ll_type == 'normal':
            if labels is not None:
                batch_y = numpyro.subsample(labels, event_dim=1)
            else:
                batch_y = None
            numpyro.sample(
                "obs", dist.Normal(pred, sigma).to_event(1), obs=batch_y
            )
        elif ll_type == 'categorical':
            if labels is not None:
                batch_y = numpyro.subsample(labels, event_dim=0)
            else:
                batch_y = None
            numpyro.sample(
                "obs", dist.Categorical(logits=pred), obs=batch_y
            )

def densenet(images, sigma=1., labels=None, subsample_size=None, likelihood_type='normal', train=False):
    n, d = images.shape

    nnet = flax_module("nnet", DenseNet([300, 100, 10], fnn.swish), input_shape=(1, d))

    likelihood(nnet, images, labels, sigma, n, subsample_size, likelihood_type)


def lenet(images, sigma=1., labels=None, subsample_size=None, likelihood_type='normal', train=False):
    n, d, _, _ = images.shape

    nnet = flax_module("nnet", LeNet(conv_features=[8, 16], dense_features=[120, 84, 10]), input_shape=(1, d, d, 1))

    likelihood(nnet, images, labels, sigma, n, subsample_size, likelihood_type, event_dim=3)


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)


def resnet(images, sigma=1., labels=None, subsample_size=None, likelihood_type='normal', train=False):
    n, d, _, _ = images.shape

    nnet = flax_module("nnet", ResNet50(num_classes=10), input_shape=(1, d, d, 1), mutable=["batch_stats"], train=True)

    likelihood(nnet, images, labels, sigma, n, subsample_size, likelihood_type, event_dim=3, train=train)

def fitting_and_testing(model, train_ds, test_ds, rng_key, likelihood_type):
    guide = lambda *args, **kwargs: None  # MLE estimate
    opt = optax.chain(
        optax.clip(100.),
        optax.adam(1e-4)
    )
    optimizer = numpyro.optim.optax_to_numpyro(opt)

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))
    #########################################

    rng_key, _rng_key = random.split(rng_key)

    if likelihood_type == 'normal':
        svi_result = svi.run(
            _rng_key, 
            10000, 
            train_ds['image'], 
            labels=nn.one_hot(train_ds['label'], 10), 
            subsample_size=256,
            likelihood_type=likelihood_type,
            train=True)
    elif likelihood_type == 'categorical':
        svi_result = svi.run(
            _rng_key, 
            10000, 
            train_ds['image'], 
            labels=train_ds['label'], 
            subsample_size=256,
            likelihood_type=likelihood_type,
            train=True)


    params, mutable_states = svi_result.params, svi_result.state.mutable_state
    if mutable_states is None:
        mutable_states = {}

    # for some reason testing takes additional memory so the results have to be casted
    # back to cpu before testing to reduce memory usage on gpu
    params = device_put(params, devices('cpu')[0])  

    pred = Predictive(model, params={**params, **mutable_states}, num_samples=1)
    rng_key, _rng_key = random.split(rng_key)
    sample = pred(_rng_key, test_ds['image'], sigma=1e-6)
    acc = jnp.mean( sample['obs'][0].argmax(-1) == test_ds['label'] )
    
    print('model acc :', acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep neural networks training")
    parser.add_argument("-n", "--network", nargs='?', default='DenseNet', type=str)
    parser.add_argument("--device", nargs='?', default='gpu', type=str)
    parser.add_argument("--seed", nargs='?', default=0, type=int)
    parser.add_argument("-ll", "--likelihood", nargs='?', default='normal', type=str)

    args = parser.parse_args()
    numpyro.set_platform(args.device)

    # load data
    train_ds, test_ds = MNIST()
    
    print('Fitting {} with {} likelihood.\n'.format(args.network, args.likelihood))

    if args.network == 'DenseNet':
        # reshape images for dense networks
        train_ds['image'] = train_ds['image'].reshape(train_ds['image'].shape[0], -1)
        test_ds['image'] = test_ds['image'].reshape(test_ds['image'].shape[0], -1)
        model = densenet
    elif args.network == 'LeNet':
        model = lenet
    
    elif args.network == 'ResNet':
        model = resnet

    rng_key = random.PRNGKey(args.seed)
    fitting_and_testing(model, train_ds, test_ds, rng_key, likelihood_type=args.likelihood)