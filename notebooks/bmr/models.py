import jax.numpy as jnp
import jax.tree_util as jtu
import numpyro.distributions as dist
import optax
import equinox as eqx

from collections import defaultdict
from functools import partial
from jax.scipy.special import gammaln
from jax import nn, lax, random, vmap, device_put, linearize
from numpyro import handlers, sample, plate, deterministic, factor, subsample, param
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO, Predictive, log_likelihood
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform, AffineTransform, ComposeTransform, ExpTransform
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal, AutoBNAFNormal, AutoLaplaceApproximation
from numpyro.optim import optax_to_numpyro, Minimize
from optax_adan import adan

adabelief = lambda *args, **kwargs: optax.adabelief(*args, eps=1e-16, eps_root=1e-16, **kwargs)

def exact_blr(X, y, lam=1, mu_0=0., a_0=2., b_0=2.):
    # bayesian linear regression
    n, D = X.shape

    mu_0 = mu_0 * jnp.ones(D)
    P_0  = jnp.diag(lam * jnp.ones(D))
    S = X.T @ X
    P_n = P_0 + S
    mu_n = jnp.linalg.solve(P_n, X.T @ y + P_0 @ mu_0)

    a_n = a_0 + n/2
    b_n = b_0 + (y.T @ y + mu_0.T @ P_0 @ mu_0 - mu_n.T @ P_n @ mu_n)/2

    return mu_n, P_n, a_n, b_n

def log_ratio(X, y, params, samples, a_0=2., b_0=2.):
    # ln Q/P for the exact posterior Q
    n, D = X.shape
    beta = samples['beta']
    sigma = samples['sigma']

    # - ln P
    log_r = -dist.Normal(X @ beta, sigma).log_prob(y).sum()
    log_r -= dist.MultivariateNormal(loc=0., scale_tril=sigma * jnp.eye(D)).log_prob(beta)
    log_r -= dist.InverseGamma(a_0, b_0).log_prob(jnp.square(sigma))

    # ln Q
    a = params['a']
    b = params['b']
    L = params['L']
    mu = params['mu']
    
    log_r += dist.MultivariateNormal(mu, scale_tril=sigma * L).log_prob(beta)
    log_r += dist.InverseGamma(a, b).log_prob(jnp.square(sigma))

    return log_r


class QRTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, R, R_inv):
        if jnp.ndim(R) != 2:
            raise ValueError(
                "Only support 2-dimensional R matrix. "
            )
        self.R = R
        self.R_inv = R_inv

    def __call__(self, x):
        return jnp.squeeze(
            jnp.matmul(self.R, x[..., jnp.newaxis]), axis=-1
        )

    def _inverse(self, y):
        return jnp.squeeze(
            jnp.matmul(self.R_inv, y[..., jnp.newaxis]), axis=-1
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.broadcast_to(
            jnp.log(jnp.diagonal(self.R, axis1=-2, axis2=-1)).sum(-1),
            jnp.shape(x)[:-1],
        )

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.R.shape[:-1])

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.R.shape[:-1])

class BayesRegression(object):

    def __init__(
        self, 
        rng_key, 
        X, 
        nnet, 
        *, 
        p0=1, 
        regtype='linear', 
        batch_size=None, 
        with_qr=False, 
        with_hyperprior=True,
        **kwargs
        ):

        self.N, self.D = X.shape
        self.X = X
        self.batch_size = batch_size

        self.nnet = nnet
        self.layers = self.get_linear_layers(nnet)
        
        self.rng_key = rng_key
        self.with_qr = with_qr
        self.with_hyperprior = with_hyperprior
        self.p0 = p0
        self.type = regtype # type of the rergression problem

        # prior weight uncertanty
        gamma0 = kwargs.pop('gamma0', 1.)
        self.gamma = defaultdict(lambda: gamma0)

        # parameters for sigma's gamma prior
        self.a0 = kwargs.pop('a0', 2.)
        self.b0 = kwargs.pop('b0', 2.)
        
        if self.with_qr:
            # use QR decomposition
            self.Q, self.R = jnp.linalg.qr(X)
            self.R_inv = jnp.linalg.inv(self.R)

    def set_input(self, X, batch_size=None):
        self.N, self.D = X.shape
        self.X = X
        self.batch_size = batch_size

    def likelihood(self, nnet, target, sigma):
        with plate('data', self.N, subsample_size=self.batch_size):
            batch_x = subsample(self.X, event_dim=1)
            mu = vmap(nnet)(batch_x).squeeze()

            if target is not None:
                batch_y = subsample(target, event_dim=0)
            else:
                batch_y = target

            if self.type == 'linear':
                sample('obs', dist.Normal(mu, sigma), obs=batch_y)
            elif self.type == 'logistic':
                sample('obs', dist.Bernoulli(logits=mu), obs=batch_y)
            elif self.type == 'multinomial':
                probs = deterministic('probs', nn.softmax(mu, -1))
                sample('obs', dist.Categorical(logits=mu), obs=batch_y)

    def hyperprior(self, name, shape, layer):
        c_sqr = sample(name + '.c^2', dist.InverseGamma(2., 3.))
        i, j = shape
        tau = sample(name + '.tau', dist.HalfCauchy(1.).expand([i]).to_event(1))
        
        # if layer == 0:
        eps = sample(name + '.eps', dist.Gamma(1.))
        tau0 = eps * self.p0 / (j - self.p0)
        lam = sample(name + '.lam', dist.HalfCauchy(1.).expand([i, j]).to_event(2))
        # else:
        #     tau0 = 1.
        #     lam = jnp.ones((i, j))

        psi = tau0 * jnp.expand_dims(tau, -1) * lam
        
        gamma = deterministic(name + '.gamma', jnp.sqrt(c_sqr * psi ** 2 / (c_sqr + psi ** 2)))

        return gamma

    def prior(self, name, sigma, gamma, loc=0., with_qr=False):
        if with_qr:
            rt = QRTransform(self.R, self.R_inv)
            aff = AffineTransform(loc, sigma * gamma)
            ct = ComposeTransform([aff, rt])
            with handlers.reparam(config={name + "_theta": TransformReparam()}):
                theta = sample(
                    name + '_theta', 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand(list(gamma.shape)).to_event(2), ct)
                )

            weight = deterministic(name, rt.inv(theta))

        else:
            aff = AffineTransform(loc, sigma * gamma)
            with handlers.reparam(config={name: TransformReparam()}):
                weight = sample(
                    name, 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand(list(gamma.shape)).to_event(2), aff)
                )

        return weight

    def get_linear_layers(self, layer):
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        return [x for x in jtu.tree_leaves(layer, is_leaf=is_linear) if is_linear(x)]

    def _register_network_vars(self, sigma):
        L = len(self.layers)
        new_layers = []
        for l, layer in enumerate(self.layers):
            weight, bias = layer.weight, layer.bias
            if bias is not None:
                bias = sample(f'layer{l}.bias', dist.Normal(0., 1.).expand(list(bias.shape)).to_event(1))
                layer = eqx.tree_at(lambda x: x.bias, layer, bias)
            
            if self.with_hyperprior:
                gamma = self.hyperprior(f'layer{l}.weight', weight.shape, l)
            else:
                gamma = jnp.broadcast_to(self.gamma[f'layer{l}.weight'], weight.shape)
            
            if l + 1 < L:
                if l == 0:
                    weight = self.prior(f'layer{l}.weight', 1., gamma, with_qr=self.with_qr)
                else:
                    weight = self.prior(f'layer{l}.weight', 1., gamma)
            else:
                if l == 0:
                    weight = self.prior(f'layer{l}.weight', sigma, gamma, with_qr=self.with_qr)
                else:
                    weight = self.prior(f'layer{l}.weight', sigma, gamma)
            
            layer = eqx.tree_at(lambda x: x.weight, layer, weight)
            new_layers.append(layer)
            
            if l == 0:
                beta = weight
            else:
                dim1 = weight.shape[-1]
                dim2 = beta.shape[0]
                if dim1 == dim2:
                    beta = weight @ beta
                else:
                    beta = weight[:, :-1] @ beta

        deterministic('beta', beta.squeeze())

        nnet = eqx.tree_at(self.get_linear_layers, self.nnet, new_layers)

        return nnet

    def model(self, obs=None):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', dist.Gamma(self.a0, self.b0))
            sigma = deterministic('sigma', 1/jnp.sqrt(sigma_sqr_inv))
        else:
            sigma = deterministic('sigma', jnp.ones(1))

        nnet = self._register_network_vars(sigma)
        
        self.likelihood(nnet, obs, sigma)
    
    def fit(self, data, num_samples=1000, warmup_steps=1000, num_chains=1, summary=False, progress_bar=True):
        self.rng_key, _rng_key = random.split(self.rng_key)

        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(
            nuts_kernel, 
            num_warmup=warmup_steps, 
            num_samples=num_samples, 
            num_chains=num_chains,
            chain_method='vectorized',
            progress_bar=progress_bar
        )
        
        mcmc.run(_rng_key, obs=data)

        if summary:
            mcmc.print_summary()

        samples = mcmc.get_samples(group_by_chain=False)
        self.mcmc = mcmc
        self.samples = samples

        return samples


class SVIRegression(BayesRegression):

    def __init__(
        self, 
        rng_key, 
        X, 
        nnet, 
        *,
        optimizer=adabelief,
        p0=1, 
        regtype='linear',
        batch_size=None, 
        with_qr=False, 
        with_hyperprior=True,
        **kwargs
    ):
        super().__init__(
            rng_key, 
            X, 
            nnet, 
            p0=p0, 
            regtype=regtype, 
            batch_size=batch_size, 
            with_qr=with_qr, 
            with_hyperprior=with_hyperprior,
            **kwargs
        )
        self.optimizer = optimizer
 
    def hyperprior(self, name, shape, layer):
        c_sqr_inv = sample(name + '.c^-2', dist.Gamma(2., 3.))
        i, j = shape

        p0 = min(self.p0, j - 1)

        u = sample(name + '.u', dist.Gamma(1/2, 1).expand([i, j + 1]).to_event(2))
        v = sample(name + '.v', dist.Gamma(1/2, 1).expand([i, j + 1]).to_event(2))

        if layer == 0:
            eps = sample(name + '.eps', dist.Gamma(1.))
            tau0 = eps * p0 / (j - p0)
        else:
            tau0 = 1.
        
        psi = tau0 ** 2 * v[:, 1:] * v[:, :1]
        ksi = u[:, 1:] * u[:, :1]

        gamma = deterministic(name + '.gamma', jnp.sqrt(psi / (ksi + c_sqr_inv * psi)))
        
        return gamma

    def fit(
        self, 
        data, 
        num_samples=1000, 
        num_steps=1000, 
        num_particles=10, 
        progress_bar=False, 
        opt_kwargs={'learning_rate': 1e-3}, 
        autoguide=None, 
        rank=2
        ):
        optimizer = optax_to_numpyro(self.optimizer(**opt_kwargs))
        model = self.model

        if autoguide == 'mean-field':
            guide = AutoNormal(self.model)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        elif autoguide == 'multivariate':
            guide = AutoMultivariateNormal(self.model)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        elif autoguide == 'lowrank-multivariate':
            guide = AutoLowRankMultivariateNormal(self.model, rank=rank)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        else:
            guide = AutoDelta(self.model)
            loss = TraceMeanField_ELBO(num_particles=num_particles)

        self.rng_key, _rng_key = random.split(self.rng_key)

        svi = SVI(model, guide, optimizer, loss)

        self.results = svi.run(_rng_key, num_steps, progress_bar=progress_bar, obs=data)

        self.rng_key, _rng_key = random.split(self.rng_key)
        samples = Predictive(guide, params=self.results.params, num_samples=num_samples)(_rng_key)
        
        pred = Predictive(model, posterior_samples=samples)

        self.rng_key, _rng_key = random.split(self.rng_key)
        self.samples = pred(_rng_key, obs=data)
        self.samples.update(samples)

        return self.samples

class BMRRegression(SVIRegression):

    def __init__(
        self, 
        rng_key, 
        X, 
        nnet, 
        *,
        optimizer=adabelief, 
        p0=1, 
        regtype='linear', 
        with_qr=False,
        batch_size=None, 
        posterior='normal',
        **kwargs
        ):
        super().__init__(
            rng_key, 
            X, 
            nnet, 
            optimizer=optimizer, 
            p0=p0, 
            regtype=regtype, 
            with_qr=with_qr,
            batch_size=batch_size, 
            with_hyperprior=False,
            **kwargs
        )
        self.posterior = posterior

    def prior(self, name, sigma, gamma, value=0., with_qr=False):
        if with_qr:
            raise NotImplementedError
        else:
            aff = AffineTransform(value, sigma)
            with handlers.reparam(config={name: TransformReparam()}):
                weight = sample(
                    name, 
                    dist.TransformedDistribution(dist.Normal(0., gamma).to_event(2), aff)
                )

        return weight
    
    def normal_weight_posterior(self, name, sigma, shape):
        if self.with_qr:
            raise NotImplementedError
        else:
            loc = param(name + '.loc', lambda rng_key: random.normal(rng_key, shape=shape) / 10)
            scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.positive)
            sample(name + '_base', dist.Normal(loc/sigma, scale).to_event(2))

    def multivariate_weight_posterior(self, name, sigma, shape):
        if self.with_qr:
            raise NotImplementedError
        else:
            loc = param(name + '.loc', lambda rng_key: random.normal(rng_key, shape=shape) / 10)
            scale = param(name + '.scale', vmap(jnp.diag)(jnp.ones(shape)), constraint=constraints.scaled_unit_lower_cholesky)
            sample(name + '_base', dist.MultivariateNormal(loc/sigma, scale_tril=scale).to_event(1))

    def get_weight_posterior(self, name, sigma, shape):
        if self.posterior == 'normal':
            return self.normal_weight_posterior(name, sigma, shape)

        elif self.posterior == 'multivariate':
            return self.multivariate_weight_posterior(name, sigma, shape)

        else:
            raise NotImplementedError

    def __lognormal(self, name, shape):
        loc = param(name + '.loc', lambda rng_key: random.normal(rng_key, shape=shape))
        scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.positive)
        return dist.LogNormal(loc, scale)

    def guide(self, obs=None):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', self.__lognormal('sigma^-2', (1,)))
            sigma = 1/jnp.sqrt(sigma_sqr_inv)
        else:
            sigma = jnp.ones(1)

        L = len(self.layers)
        for l, layer in enumerate(self.layers):
            if layer.bias is not None:
                loc = param(f'layer{l}.bias.loc', lambda rng_key: random.normal(rng_key, shape=layer.bias.shape) / 10)
                scale = param(f'layer{l}.bias.scale', jnp.ones(layer.bias.shape)/10, constraint=constraints.positive)
                sample(f'layer{l}.bias', dist.Normal(loc, scale).to_event(1))
            
            if l + 1 < L:
                self.get_weight_posterior(f'layer{l}.weight', 1., layer.weight.shape)
            else:
                self.get_weight_posterior(f'layer{l}.weight', sigma, layer.weight.shape) 

    def ΔF(self, mu, P, gamma, sigma_sqr=1):
        if 'normal' in self.posterior:
            return self.ΔF_mf(mu, P, gamma, sigma_sqr)
        elif self.posterior == 'multivariate':
            return self.ΔF_mv(mu, P, gamma, sigma_sqr)
        else:
            raise NotImplementedError

    def ΔF_mv(self, mu, P, gamma_sqr, sigma_sqr):

        M = jnp.diag(gamma_sqr) @ P + jnp.diag(1 - gamma_sqr/sigma_sqr)

        _, logdet = jnp.linalg.slogdet(M)
        df = -logdet/2

        _, logdet = jnp.linalg.slogdet(P)
        df += logdet / 2

        df += jnp.sum(jnp.log(sigma_sqr))
        
        t_P = P + jnp.diag(1/gamma_sqr - 1/sigma_sqr)
        t_mu = jnp.linalg.solve(t_P, P @ mu)

        df -= mu.T @ P @ mu / 2
        df += t_mu.T @ t_P @ t_mu / 2
        
        return df, t_mu, t_P

    def ΔF_mf(self, mu, pi, gamma_sqr, sigma_sqr):
        M = jnp.clip(gamma_sqr * (pi - 1/sigma_sqr) + 1, a_min=1e-16)

        df = -jnp.log(M).sum() / 2

        df += (jnp.log(pi) + jnp.log(sigma_sqr)).sum() / 2
        
        t_pi = pi + 1/gamma_sqr - 1/sigma_sqr
        t_mu = pi * mu / t_pi

        df -= jnp.inner(pi * mu, mu) / 2
        df += jnp.inner(t_pi * t_mu, t_mu) / 2
        
        return df, t_mu, t_pi

    def sufficient_stats(self, name, invert=[]):
        '''Multivariate normal guide'''
        if len(invert) == 0:
            if self.posterior == 'normal':
                params = self.results.params
                pi = 1 / params[name + '.scale'] ** 2
                mu = params[name + '.loc']
                return (mu, pi)

            elif self.posterior == 'multivariate':
                params = self.results.params
                mu = params[name + '.loc']
                L_inv = vmap(jnp.linalg.inv)(params[name + '.scale'])
                P = jnp.matmul(L_inv.transpose((0, -1, -2)), L_inv)
                return (mu, P)

            else:
                raise NotImplementedError
        
        else:
            if self.posterior == 'normal':
                deterministic(f'{name}.loc', invert[0])
                deterministic(f'{name}.scale',  1/jnp.sqrt(invert[1]))

            elif self.posterior == 'multivariate':
                deterministic(f'{name}.loc', invert[0])
                deterministic(f'{name}.scale', vmap(lambda x: jnp.linalg.inv(jnp.linalg.cholesky(x)).T)(invert[1]))
            
            else:
                raise NotImplementedError

    def pruning(self):
        for l, layer in enumerate(self.layers):
            key = f'layer{l}.weight'
            gamma = self.hyperprior(key, layer.weight.shape, l)
            mu_n, P_n = self.sufficient_stats(key)
            sigma_sqr =  jnp.broadcast_to(jnp.square(self.gamma[key]), layer.weight.shape)
            
            log_prob, t_mu, t_P = vmap(self.ΔF)(mu_n, P_n, jnp.square(gamma), sigma_sqr)
            factor(f'layer{l}.weight.log_prob', log_prob.sum())

            self.sufficient_stats(key, invert=[t_mu, t_P])
                    
    def fit(self, data, num_samples=1000, num_steps=1000, num_particles=10, progress_bar=True, opt_kwargs={'learning_rate': 1e-4}):
        optimizer = optax_to_numpyro(optax.chain(self.optimizer(**opt_kwargs)))
        model = self.model
        if self.posterior == 'delta':
            guide = AutoDelta(self.model)
            loss = TraceMeanField_ELBO(num_particles=1)
        else:
            guide = self.guide
            loss = TraceGraph_ELBO(num_particles=num_particles)
        
        self.rng_key, _rng_key = random.split(self.rng_key)

        svi = SVI(model, guide, optimizer, loss)

        self.results = svi.run(_rng_key, num_steps, progress_bar=progress_bar, obs=data)
        
        self.rng_key, _rng_key = random.split(self.rng_key)
        samples = Predictive(guide, params=self.results.params, num_samples=num_samples)(_rng_key)
        
        pred = Predictive(model, posterior_samples=samples)

        self.rng_key, _rng_key = random.split(self.rng_key)
        self.samples = pred(_rng_key, obs=data)
        self.samples.update(samples)

    def bmr(
        self, 
        autoguide, 
        num_steps=1000, 
        num_particles=10, 
        num_samples=1000, 
        progress_bar=True, 
        opt_kwargs={'learning_rate': 1e-3}, 
        **kwargs
    ):
        optimizer = optax_to_numpyro(optax.chain(self.optimizer(**opt_kwargs)))

        if autoguide == 'mean-field':
            guide = AutoNormal(self.pruning)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        elif autoguide == 'multivariate':
            guide = AutoMultivariateNormal(self.pruning)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        elif autoguide == 'lowrank-multivariate':
            rank = kwargs.pop('rank', 2)
            guide = AutoLowRankMultivariateNormal(self.pruning, rank=rank)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        else:
            guide = AutoDelta(self.pruning)
            loss = TraceMeanField_ELBO(num_particles=num_particles)

        self.rng_key, _rng_key = random.split(self.rng_key)
        svi = SVI(self.pruning, guide, optimizer, loss)

        results = svi.run(_rng_key, num_steps, progress_bar=progress_bar, stable_update=True)

        pred = Predictive(self.pruning, guide=guide, params=results.params, num_samples=num_samples)

        self.rng_key, _rng_key = random.split(self.rng_key)
        samples = pred(_rng_key)

        for key in samples:
            if 'gamma' in key:
                s = key.split('.')
                s = s[0] + '.' + s[1]
                self.gamma[s] = jnp.sqrt(jnp.square(samples[key]).mean(0))

        loc_params = self.results.params.copy()
        extra = {}
        for key in loc_params:
            if 'weight' in key:
                loc_params[key] = samples[key].mean(0)
        
        loc_params.update(extra)

        self.rng_key, _rng_key = random.split(self.rng_key)
        pred = Predictive(self.guide, params=loc_params, num_samples=num_samples)
        samples = pred(_rng_key)

        self.rng_key, _rng_key = random.split(self.rng_key)
        self.samples = Predictive(self.model, posterior_samples=samples)(_rng_key)
        self.samples.update(samples)
        
        return results, self.samples


class LoopyBMR(SVIRegression):

    def __init__(
        self, 
        rng_key, 
        X, 
        nnet, 
        *, 
        p0=1, 
        regtype='linear', 
        with_qr=False, 
        posterior='normal'
    ):
        super().__init__(
            rng_key, 
            X, 
            nnet, 
            p0=p0, 
            regtype=regtype, 
            with_qr=with_qr, 
            with_hyperprior=False, 
            posterior=posterior
        )

    def fit(self, data, num_samples=1000, num_steps=1000, num_particles=1, progress_bar=True, opt_kwargs={'learning_rate': 1e-3}):
        optimizer = optax_to_numpyro(optax.chain(optax.adabelief(**opt_kwargs)))
        model = self.model
        guide = self.guide

        loss = Trace_ELBO(num_particles=num_particles)
        self.rng_key, _rng_key = random.split(self.rng_key)

        svi = SVI(model, guide, optimizer, loss)
        # params = f()
        init_state = svi.init(_rng_key, obs=data, params=params)

        loss = TraceMeanField_ELBO(num_particles=num_particles)

        self.rng_key, _rng_key = random.split(self.rng_key)
        svi2 = SVI(self.pruning, AutoNormal(self.pruning), optimizer, loss)
        # params = h(init_state)
        init_hyper = svi2.init(_rng_key, params=params)

        for i in range(10):
            init_state, loss = svi.update(init_state, obs=data)

        self.results = svi.run(_rng_key, num_steps, progress_bar=progress_bar, obs=data)
        
        pred = Predictive(
            model, 
            guide=guide, 
            params=state.params, 
            num_samples=num_samples
        )
        
        self.rng_key, _rng_key = random.split(self.rng_key)
        self.samples = pred(_rng_key, obs=data)

        return self.samples


# TODO: implement BMR for laplace approximation
# def linearize_nnet(nnet, noise, x):
#     params, static = eqx.partition(nnet, eqx.is_inexact_array)
#     def f(p):
#         nn = eqx.combine(p, static)
#         return nn(x)

#     _, f_jvp = linearize(f, lax.stop_gradient(params))

#     return nnet(x) + f_jvp(noise)

# class LinearizedRegression(SVIRegression):

#     def __init__(
#         self, 
#         rng_key, 
#         X, 
#         nnet, 
#         *,
#         optimizer=adan,
#         p0=1, 
#         regtype='linear', 
#         with_qr=False, 
#         with_hyperprior=True,
#         posterior='multivariate'
#     ):
#         super().__init__(rng_key, X, nnet, p0=p0, regtype=regtype, with_qr=with_qr, with_hyperprior=with_hyperprior, optimizer=optimizer)
#         self.posterior=posterior
 
#     def rnp(self):
#         # registerm network parameters
#         new_vals = ()
#         L = len(self.vals[0])
#         for l, layer in enumerate(self.vals[0]):
#             lv, l_aux = layer.tree_flatten()
#             new_lv = ()
#             for value, name in zip(lv, l_aux[0]):
#                 if value is not None:
#                     new_lv += (param(f'layer{l}.{name}.loc', lambda rng_key: random.normal(rng_key, shape=value.shape) / 10),)
#                 else:
#                     new_lv += (value,)

#             new_vals += (layer.tree_unflatten(l_aux, new_lv), )

#         vals = (new_vals,) + self.vals[1:]
#         params = self.params.tree_unflatten(self.aux, vals)

#         return eqx.combine(params, self.static)

#     def rrnp(self, nnet, sigma):
#         # register network params as random variables
#         params, static = eqx.partition(nnet, eqx.is_inexact_array)
#         vals, aux = params.tree_flatten()
#         new_vals = ()
#         L = len(vals[0])
#         for l, layer in enumerate(vals[0]):
#             lv, l_aux = layer.tree_flatten()
#             new_lv = ()
#             for value, name in zip(lv, l_aux[0]):
#                 if value is not None:
#                     if name == 'bias':
#                         new_lv += (sample(f'layer{l}.{name}', dist.Normal(value, 1.).to_event(1)),)
#                     else:
#                         if self.with_hyperprior:
#                             gamma = self.hyperprior(f'layer{l}.{name}', value.shape)
#                         else:
#                             gamma = jnp.ones(value.shape)
#                         if l + 1 < L:
#                             weight = self.prior(f'layer{l}.{name}', 1., gamma, loc=value)
#                             log_factor = - 0.5 * jnp.square(value).sum()
#                             factor(f'layer{l}.{name}.correction', log_factor)
#                         else:
#                             weight = self.prior(f'layer{l}.{name}', sigma, gamma, loc=value)
#                             log_factor = - 0.5 * jnp.square(value).sum()/sigma ** 2
#                             factor(f'layer{l}.{name}.correction', log_factor)

#                         new_lv += (weight - value,)
#                         if l == 0:
#                             beta = weight
#                         else:
#                             beta = weight @ beta
#                 else:
#                     new_lv += (value,)

#             new_vals += (layer.tree_unflatten(l_aux, new_lv), )

#         deterministic('beta', beta.squeeze())

#         return params.tree_unflatten(aux, (new_vals,) + vals[1:])

#     def model(self, obs=None):
#         if self.type == 'linear':
#             sigma_sqr_inv = sample('sigma^-2', dist.Gamma(2., 2.))
#             sigma = deterministic('sigma', 1/jnp.sqrt(sigma_sqr_inv))
#         else:
#             sigma = deterministic('sigma', jnp.ones(1))

#         nnet = self.rnp()
#         noise = self.rrnp(nnet, sigma)
#         linnet = partial(linearize_nnet, nnet, noise)
#         mu = vmap(linnet)(self.X).squeeze()
        
#         with handlers.condition(data={'obs': obs}):
#             self.likelihood(mu, sigma)

#     def __lognormal(self, name, shape):
#         loc = param(name + '.loc', lambda rng_key: random.normal(rng_key, shape=shape))
#         scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.softplus_positive)
#         return dist.LogNormal(loc, scale)

#     def hyperposterior(self, name, shape):
#         i, j = shape
#         sample(name + '.c^-2', self.__lognormal(name + '.c^-2', (1,)))
#         sample(name + '.u', self.__lognormal(name + '.u', (i, j+1)).to_event(2))
#         sample(name + '.v', self.__lognormal(name + '.v', (i, j+1)).to_event(2))
#         # sample(name + '.eps', self.__lognormal(name + '.eps', (1,)))

#     def normal_weight_posterior(self, name, shape):
#         if self.with_qr:
#             raise NotImplementedError
#         else:
#             scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.softplus_positive)
#             sample(name + '_base', dist.Normal(0., scale).to_event(2))

#     def multivariate_weight_posterior(self, name, shape):
#         if self.with_qr:
#             raise NotImplementedError
#         else:
#             scale = param(name + '.scale', vmap(jnp.diag)(jnp.ones(shape))/10, constraint=constraints.scaled_unit_lower_cholesky)
#             sample(name + '_base', dist.MultivariateNormal(0., scale_tril=scale).to_event(1))

#     def weight_posterior(self, name, shape):
#         if self.posterior == 'normal':
#             return self.normal_weight_posterior(name, shape)
#         elif self.posterior == 'multivariate':
#             return self.multivariate_weight_posterior(name, shape)
#         else:
#             raise NotImplementedError
        
#     def guide(self, obs=None):
#         if self.type == 'linear':
#             sigma_sqr_inv = sample('sigma^-2', self.__lognormal('sigma^-2', (1,)))

#         L = len(self.vals[0])
#         for l, layer in enumerate(self.vals[0]):
#             lv, l_aux = layer.tree_flatten()
#             new_lv = ()
#             for value, name in zip(lv, l_aux[0]):
#                 if value is not None:
#                     if name == 'bias':
#                         scale = param(f'layer{l}.{name}.scale', jnp.ones(value.shape)/10, constraint=constraints.softplus_positive)
#                         sample(f'layer{l}.{name}_base', dist.Normal(0., scale).to_event(1))
#                     else:
#                         if self.with_hyperprior:
#                             self.hyperposterior(f'layer{l}.{name}', value.shape)

#                         if l + 1 < L:
#                             self.weight_posterior(f'layer{l}.{name}', value.shape)
#                         else:
#                             self.weight_posterior(f'layer{l}.{name}', value.shape)

#     def fit(
#         self, 
#         data, 
#         num_samples=1000, 
#         num_steps=1000, 
#         num_particles=10, 
#         progress_bar=True, 
#         opt_kwargs={'learning_rate': 1e-3}, 
#         autoguide=None, 
#         rank=2
#         ):
#         optimizer = optax_to_numpyro(self.optimizer(**opt_kwargs))
#         model = self.model

#         if autoguide == 'mean-field':
#             guide = AutoNormal(self.model)
#         elif autoguide == 'multivariate':
#             guide = AutoMultivariateNormal(self.model)
#         elif autoguide == 'lowrank-multivariate':
#             guide = AutoLowRankMultivariateNormal(self.model, rank=rank)
#         elif autoguide == 'bnaf-normal':
#             guide = AutoBNAFNormal(self.model)
#         elif autoguide == 'test':
#             guide = self.guide
#         else:
#             guide = AutoDelta(self.model)
            
#         loss = TraceGraph_ELBO(num_particles=num_particles)
#         self.rng_key, _rng_key = random.split(self.rng_key)

#         svi = SVI(model, guide, optimizer, loss)

#         self.results = svi.run(_rng_key, num_steps, progress_bar=progress_bar, obs=data)
        
#         pred = Predictive(model, guide=guide, params=self.results.params, num_samples=num_samples)
        
#         self.samples = pred(_rng_key, obs=data)

#         return self.samples