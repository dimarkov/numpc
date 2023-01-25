import jax.numpy as jnp
import jax.tree_util as jtu
import numpyro.distributions as dist
import optax
import equinox as eqx

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
        with_hyperprior=True
        ):

        self.N, self.D = X.shape
        self.X = X
        self.batch_size = batch_size
        self.params, self.static = eqx.partition(nnet, eqx.is_inexact_array)
        self.vals, self.aux = self.params.tree_flatten()

        self.rng_key = rng_key
        self.with_qr = with_qr
        self.with_hyperprior = with_hyperprior
        self.p0 = p0
        self.type = regtype # type of the rergression problem

        # parameters for sigma's gamma prior
        self.a0 = 2.
        self.b0 = 2.
        
        if self.with_qr:
            # use QR decomposition
            self.Q, self.R = jnp.linalg.qr(X)
            self.R_inv = jnp.linalg.inv(self.R)

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
                sample('obs', dist.Categorical(logits=mu), obs=batch_y)

    def hyperprior(self, name, shape, use=True):
        if use:
            c_sqr = sample(name + '.c^2', dist.InverseGamma(2., 3.))
            i, j = shape
            tau = sample(name + '.tau', dist.HalfCauchy(1.).expand([i]).to_event(1))
            lam = sample(name + '.lam', dist.HalfCauchy(1.).expand([i, j]).to_event(2))

            eps = 1. # sample(name + '.eps', dist.Exponential(jnp.sqrt(self.N)) )
            tau0 = eps * self.p0 / (j - self.p0)

            psi = tau0 * jnp.expand_dims(tau, -1) * lam
            
            gamma = deterministic(name + '.gamma^2', jnp.sqrt(c_sqr * psi ** 2 / (c_sqr + psi ** 2)))

            return gamma
        else:
            return jnp.ones(1)

    def prior(self, name, sigma, gamma, loc=0.):
        if self.with_qr:
            #TODO: test if it is working
            raise NotImplementedError
            # rt = QRTransform(self.R, self.R_inv)
            # aff = AffineTransform(0., gamma)
            # ct = ComposeTransform([aff, rt])
            # with handlers.reparam(config={name + "_theta": TransformReparam()}):
            #     theta = sample(
            #         name + '_theta', 
            #         dist.TransformedDistribution(dist.Normal(0., 1.).expand(list(shape)).to_event(2), ct)
            #     )

            # weight = deterministic(name, rt.inv(theta))

        else:
            aff = AffineTransform(loc, sigma * gamma)
            with handlers.reparam(config={name: TransformReparam()}):
                weight = sample(
                    name, 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand(list(gamma.shape)).to_event(2), aff)
                )

        return weight

    def _register_network_vars(self, sigma):
        new_vals = ()
        L = len(self.vals[0])
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        new_lv += (sample(f'layer{l}.{name}', dist.Normal(0., 1.).expand(list(value.shape)).to_event(1)),)
                    else:
                        if self.with_hyperprior:
                            gamma = self.hyperprior(f'layer{l}.{name}', value.shape)
                        else:
                            gamma = jnp.ones(value.shape)
                        
                        if l + 1 < L:
                            weight = self.prior(f'layer{l}.{name}', 1., gamma)
                        else:
                            weight = self.prior(f'layer{l}.{name}', sigma, gamma)
                        
                        new_lv += (weight,)
                        if l == 0:
                            beta = weight
                        else:
                            beta = weight @ beta
                else:
                    new_lv += (value,)

            new_vals += (layer.tree_unflatten(l_aux, new_lv), )

        deterministic('beta', beta.squeeze())

        vals = (new_vals,) + self.vals[1:]
        params = self.params.tree_unflatten(self.aux, vals)

        return eqx.combine(params, self.static)
        
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
        optimizer=adan,
        p0=1, 
        regtype='linear', 
        with_qr=False, 
        with_hyperprior=True,
    ):
        super().__init__(rng_key, X, nnet, p0=p0, regtype=regtype, with_qr=with_qr, with_hyperprior=with_hyperprior)
        self.optimizer = optimizer
 
    def hyperprior(self, name, shape):
        c_sqr_inv = sample(name + '.c^-2', dist.Gamma(2., 3.))
        i, j = shape

        p0 = min(self.p0, j - 1)

        u = sample(name + '.u', dist.Gamma(1/2, 1).expand([i, j + 1]).to_event(2))
        v = sample(name + '.v', dist.Gamma(1/2, 1).expand([i, j + 1]).to_event(2))

        eps = 1. # sample(name + '.eps', dist.Exponential(1.))
        tau0 = eps * p0 / (j - p0)
        
        psi = tau0 ** 2 * v[:, 1:] * v[:, :1]
        ksi = u[:, 1:] * u[:, :1]

        gamma = deterministic(name + '.gamma^2', jnp.sqrt(psi / (ksi + c_sqr_inv * psi)))
        
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
        elif autoguide == 'multivariate':
            guide = AutoMultivariateNormal(self.model)
        elif autoguide == 'lowrank-multivariate':
            guide = AutoLowRankMultivariateNormal(self.model, rank=rank)
        elif autoguide == 'bnaf-normal':
            guide = AutoBNAFNormal(self.model)
        elif autoguide == 'test':
            model = self.model_test
            guide = self.guide_test
        else:
            guide = AutoDelta(self.model)
            
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

        return self.samples

class BMRRegression(SVIRegression):

    def __init__(
        self, 
        rng_key, 
        X, 
        nnet, 
        *,
        optimizer=adan, 
        p0=1, 
        regtype='linear', 
        with_qr=False, 
        posterior='normal'
        ):
        super().__init__(
            rng_key, 
            X, 
            nnet, 
            optimizer=optimizer, 
            p0=p0, 
            regtype=regtype, 
            with_qr=with_qr, 
            with_hyperprior=False
        )
        self.posterior = posterior


    def prior(self, name, sigma, gamma, value=0.):
        if self.with_qr:
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
            scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.softplus_positive)
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
        scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.softplus_positive)
        return dist.LogNormal(loc, scale)

    def guide(self, obs=None):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', self.__lognormal('sigma^-2', (1,)))
            sigma = 1/jnp.sqrt(sigma_sqr_inv)
        else:
            sigma = jnp.ones(1)

        L = len(self.vals[0])
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        loc = param(f'layer{l}.{name}.loc', lambda rng_key: random.normal(rng_key, shape=value.shape) / 10)
                        scale = param(f'layer{l}.{name}.scale', jnp.ones(value.shape)/10, constraint=constraints.softplus_positive)
                        sample(f'layer{l}.{name}_base', dist.Normal(loc, scale).to_event(1))
                    else:
                        if l + 1 < L:
                            self.get_weight_posterior(f'layer{l}.{name}', 1., value.shape)
                        else:
                            self.get_weight_posterior(f'layer{l}.{name}', sigma, value.shape) 

    def ΔF(self, mu, P, gamma):
        if self.posterior == 'normal':
            return self.ΔF_mf(mu, P, gamma)
        elif self.posterior == 'multivariate':
            return self.ΔF_mv(mu, P, gamma)
        else:
            raise NotImplementedError

    def ΔF_mv(self, mu, P, gamma):
        M = jnp.diag(gamma) @ P + jnp.diag(1 - gamma)

        _, logdet = jnp.linalg.slogdet(M)
        df = -logdet/2

        _, logdet = jnp.linalg.slogdet(P)
        df += logdet / 2
        
        t_P = P + jnp.diag(1/gamma - 1)
        t_mu = jnp.linalg.solve(t_P, P @ mu)

        df -= mu.T @ P @ mu / 2
        df += t_mu.T @ t_P @ t_mu / 2
        
        return df, t_mu, t_P

    def ΔF_mf(self, mu, pi, gamma):
        M = gamma * pi + 1 - gamma

        df = -jnp.log(M).sum() / 2

        df += jnp.log(pi).sum() / 2
        
        t_pi = pi + 1/gamma - 1
        t_mu = pi * mu / t_pi

        df -= jnp.inner(pi * mu, mu) / 2
        df += jnp.inner(t_pi * t_mu, t_mu) / 2
        
        return df, t_mu, t_pi

    def sufficient_stats(self, name, invert=[]):
        '''Multivariate normal guide'''
        if len(invert) == 0:
            if self.posterior == 'normal':
                params = self.results.params
                pi = 1/params[name + '.scale'] ** 2
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
            return None

    def pruning(self):
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'weight':
                        gamma = deterministic(f'layer{l}.{name}.gamma', self.hyperprior(f'layer{l}.{name}', value.shape))
                        mu_n, P_n = self.sufficient_stats(f'layer{l}.{name}')
                        log_prob, t_mu, t_P = vmap(self.ΔF)(mu_n, P_n, gamma)
                        factor(f'layer{l}.{name}.log_prob', log_prob.sum())
                        self.sufficient_stats(f'layer{l}.{name}', invert=[t_mu, t_P])
                    
    def fit(self, data, num_samples=1000, num_steps=1000, num_particles=10, progress_bar=True, opt_kwargs={'learning_rate': 1e-3}):
        optimizer = optax_to_numpyro(optax.chain(self.optimizer(**opt_kwargs)))
        model = self.model
        guide = self.guide

        if self.posterior == 'normal':
            loss = TraceMeanField_ELBO(num_particles=num_particles)
        else:
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
        
        return self.samples

    def bmr(self, autoguide, num_steps=1000, num_particles=10, rank=2, progress_bar=True, opt_kwargs={'learning_rate': 1e-3}):
        num_samples = 1000
        optimizer = optax_to_numpyro(optax.chain(self.optimizer(**opt_kwargs)))

        if autoguide == 'mean-field':
            guide = AutoNormal(self.pruning)
            loss = TraceMeanField_ELBO(num_particles=num_particles)
        elif autoguide == 'multivariate':
            guide = AutoMultivariateNormal(self.pruning)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        elif autoguide == 'lowrank-multivariate':
            guide = AutoLowRankMultivariateNormal(self.pruning, rank=rank)
            loss = TraceGraph_ELBO(num_particles=num_particles)
        else:
            guide = AutoDelta(self.pruning)
            loss = TraceGraph_ELBO(num_particles=num_particles)

        self.rng_key, _rng_key = random.split(self.rng_key)
        svi = SVI(self.pruning, guide, optimizer, loss)

        results = svi.run(_rng_key, num_steps, progress_bar=progress_bar)

        pred = Predictive(self.pruning, guide=guide, params=results.params, num_samples=num_samples)

        self.rng_key, _rng_key = random.split(self.rng_key)
        samples = pred(_rng_key)

        loc_params = self.results.params.copy()
        for key in loc_params:
            if 'weight' in key:
                loc_params[key] = samples[key].mean(0)

        self.rng_key, _rng_key = random.split(self.rng_key)
        samples = Predictive(self.guide, params=loc_params, num_samples=num_samples)(_rng_key)
        
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


class BayesDNN(object):
    def __init__(self, rng_key, nnet, images, tau_0=1, subsample_size=64):
        self.N, self.D = images.shape
        self.images = images

        self.rng_key = rng_key
        self.tau_0 = tau_0

        self.nnet = nnet
        self.subsample_size=subsample_size

        self.params, self.static = eqx.partition(nnet, eqx.is_inexact_array)
        self.vals, self.aux = self.params.tree_flatten()

        self.rank = 2

    def likelihood(self, nnet, labels):
        with plate("N", self.N, subsample_size=self.subsample_size):
            batch_x = subsample(self.images, event_dim=1)
            pred = nnet(batch_x)
            
            if labels is not None:
                batch_y = subsample(labels, event_dim=0)
            else:
                batch_y = None
            
            sample(
                "obs", dist.Categorical(logits=pred), obs=batch_y
            )

    def _register_network_vars(self):
        new_vals = []
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        new_lv += (sample(f'layer{l}.{name}', dist.Normal(0., 10.).expand(list(value.shape)).to_event(1)),)
                    else:
                        new_lv += (sample(f'layer{l}.{name}', dist.Normal(0., 1.).expand(list(value.shape)).to_event(1)),)
                else:
                    new_lv += (value,)

            new_vals.append( layer.tree_unflatten(l_aux, new_lv) )

        vals = (new_vals,) + self.vals[1:]
        params = self.params.tree_unflatten(self.aux, vals)

        return eqx.combine(params, self.static)

    def model(self, labels=None):
        nnet = vmap(self._register_network_vars())
        self.likelihood(nnet, labels)

    def lowrank_guide(self, labels=None):
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        name = f'layer{l}.{name}'
                        loc = param('loc.' + name, value)
                        scale = param('scale.' + name, jnp.diag(jnp.ones(value.shape))/10, constraint=dist.constraints.lower_cholesky)
                        sample(name, dist.MultivariateNormal(loc, scale_tril=scale))
                    else:
                        name = f'layer{l}.{name}'
                        loc = param('loc.' + name, value)
                        i, j = value.shape
                        cov_diag = param('cov_diag.' + name, jnp.ones((i, j))/10, constraint=dist.constraints.positive)
                        cov_factor = param('cov_factor.' + name, jnp.zeros((i, j, self.rank)))
                        sample(name, dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag))
        
    def gamma(self, c_sqr, v, u):
        return self.tau_0**2 * v[..., 1:] * v[..., 0] / (c_sqr * u[..., 1:] * u[..., 0] + self.tau_0**2 * v[..., 1:] * v[..., 0])

    def ΔF(self, mu, Sigma, gamma):
        
        M = jnp.diag(gamma) + Sigma * jnp.expand_dims(1 - gamma, -2)

        _, logdet = jnp.linalg.slogdet(M)
        df = - logdet/2
        
        df += jnp.dot(mu, jnp.linalg.solve(Sigma, jnp.expand_dims(gamma, -1) * jnp.linalg.solve(M, mu)/2))

        return df
            
    def bmr_model(self, params):
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'weight':
                        name = f'layer{l}.{name}'
                        shape = value.shape
                        c_sqr = sample(f'{name}.c^2', dist.Gamma(2, 1))
                        with plate(name, shape[0]):
                            u = sample(f'{name}.u', dist.Gamma(1/2, 1).expand([shape[-1] + 1]).to_event(1))
                            v = sample(f'{name}.v', dist.Gamma(1/2, 1).expand([shape[-1] + 1]).to_event(1))

                            tau = deterministic(f'{name}.tau', self.tau_0 * jnp.sqrt(v[..., 0]/u[..., 0]))
                            lam = deterministic(f'{name}.lam', tau * jnp.sqrt(v[..., 1:]/u[..., 1:]))
                            g = deterministic('gamma^2', self.gamma(c_sqr, v, u))
                            
                            mu = params['loc.' + name]
                            cov_diag = params['cov_diag.' + name]
                            cov_factor = params['cov_factor.' + name]

                            Sigma = vmap(jnp.diag)(cov_diag)
                            Sigma += jnp.matmul(cov_factor, jnp.swapaxes(cov_factor, -1, -2))
                            log_prob = vmap(self.self.ΔF)(mu, Sigma, g)
                            factor(f'{name}.log_prob', log_prob)

    def run_svi(self, rng_key, num_steps, model, guide, optimizer, loss, *args, **kwargs):
        svi = SVI(model, guide, optimizer, loss)
        return svi.run(rng_key, num_steps, *args, **kwargs)

    def fit(self, labels, num_warmup=1000, num_samples=1000, num_steps=1000, num_particles=1, opt_kwargs={'learning_rate': 1e-3}, autoguide=None):

        _model = self.model
        _guide = self.lowrank_guide
        optimizer = optax_to_numpyro(optax.chain(optax.adabelief(**opt_kwargs)))

        self.rng_key, _rng_key = random.split(self.rng_key)
        result = self.run_svi(_rng_key, num_steps, _model, _guide, optimizer, Trace_ELBO(num_particles), labels=labels)

        self.losess = result.losses
        params = result.params

        samples = None
        return params


def linearize_nnet(nnet, noise, x):
    params, static = eqx.partition(nnet, eqx.is_inexact_array)
    def f(p):
        nn = eqx.combine(p, static)
        return nn(x)

    _, f_jvp = linearize(f, lax.stop_gradient(params))

    return nnet(x) + f_jvp(noise)

class LinearizedRegression(SVIRegression):

    def __init__(
        self, 
        rng_key, 
        X, 
        nnet, 
        *,
        optimizer=adan,
        p0=1, 
        regtype='linear', 
        with_qr=False, 
        with_hyperprior=True,
        posterior='multivariate'
    ):
        super().__init__(rng_key, X, nnet, p0=p0, regtype=regtype, with_qr=with_qr, with_hyperprior=with_hyperprior, optimizer=optimizer)
        self.posterior=posterior
 
    def rnp(self):
        # registerm network parameters
        new_vals = ()
        L = len(self.vals[0])
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    new_lv += (param(f'layer{l}.{name}.loc', lambda rng_key: random.normal(rng_key, shape=value.shape) / 10),)
                else:
                    new_lv += (value,)

            new_vals += (layer.tree_unflatten(l_aux, new_lv), )

        vals = (new_vals,) + self.vals[1:]
        params = self.params.tree_unflatten(self.aux, vals)

        return eqx.combine(params, self.static)

    def rrnp(self, nnet, sigma):
        # register network params as random variables
        params, static = eqx.partition(nnet, eqx.is_inexact_array)
        vals, aux = params.tree_flatten()
        new_vals = ()
        L = len(vals[0])
        for l, layer in enumerate(vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        new_lv += (sample(f'layer{l}.{name}', dist.Normal(value, 1.).to_event(1)),)
                    else:
                        if self.with_hyperprior:
                            gamma = self.hyperprior(f'layer{l}.{name}', value.shape)
                        else:
                            gamma = jnp.ones(value.shape)
                        if l + 1 < L:
                            weight = self.prior(f'layer{l}.{name}', 1., gamma, loc=value)
                            log_factor = - 0.5 * jnp.square(value).sum()
                            factor(f'layer{l}.{name}.correction', log_factor)
                        else:
                            weight = self.prior(f'layer{l}.{name}', sigma, gamma, loc=value)
                            log_factor = - 0.5 * jnp.square(value).sum()/sigma ** 2
                            factor(f'layer{l}.{name}.correction', log_factor)

                        new_lv += (weight - value,)
                        if l == 0:
                            beta = weight
                        else:
                            beta = weight @ beta
                else:
                    new_lv += (value,)

            new_vals += (layer.tree_unflatten(l_aux, new_lv), )

        deterministic('beta', beta.squeeze())

        return params.tree_unflatten(aux, (new_vals,) + vals[1:])

    def model(self, obs=None):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', dist.Gamma(2., 2.))
            sigma = deterministic('sigma', 1/jnp.sqrt(sigma_sqr_inv))
        else:
            sigma = deterministic('sigma', jnp.ones(1))

        nnet = self.rnp()
        noise = self.rrnp(nnet, sigma)
        linnet = partial(linearize_nnet, nnet, noise)
        mu = vmap(linnet)(self.X).squeeze()
        
        with handlers.condition(data={'obs': obs}):
            self.likelihood(mu, sigma)

    def __lognormal(self, name, shape):
        loc = param(name + '.loc', lambda rng_key: random.normal(rng_key, shape=shape))
        scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.softplus_positive)
        return dist.LogNormal(loc, scale)

    def hyperposterior(self, name, shape):
        i, j = shape
        sample(name + '.c^-2', self.__lognormal(name + '.c^-2', (1,)))
        sample(name + '.u', self.__lognormal(name + '.u', (i, j+1)).to_event(2))
        sample(name + '.v', self.__lognormal(name + '.v', (i, j+1)).to_event(2))
        # sample(name + '.eps', self.__lognormal(name + '.eps', (1,)))

    def normal_weight_posterior(self, name, shape):
        if self.with_qr:
            raise NotImplementedError
        else:
            scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.softplus_positive)
            sample(name + '_base', dist.Normal(0., scale).to_event(2))

    def multivariate_weight_posterior(self, name, shape):
        if self.with_qr:
            raise NotImplementedError
        else:
            scale = param(name + '.scale', vmap(jnp.diag)(jnp.ones(shape))/10, constraint=constraints.scaled_unit_lower_cholesky)
            sample(name + '_base', dist.MultivariateNormal(0., scale_tril=scale).to_event(1))

    def weight_posterior(self, name, shape):
        if self.posterior == 'normal':
            return self.normal_weight_posterior(name, shape)
        elif self.posterior == 'multivariate':
            return self.multivariate_weight_posterior(name, shape)
        else:
            raise NotImplementedError
        
    def guide(self, obs=None):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', self.__lognormal('sigma^-2', (1,)))

        L = len(self.vals[0])
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        scale = param(f'layer{l}.{name}.scale', jnp.ones(value.shape)/10, constraint=constraints.softplus_positive)
                        sample(f'layer{l}.{name}_base', dist.Normal(0., scale).to_event(1))
                    else:
                        if self.with_hyperprior:
                            self.hyperposterior(f'layer{l}.{name}', value.shape)

                        if l + 1 < L:
                            self.weight_posterior(f'layer{l}.{name}', value.shape)
                        else:
                            self.weight_posterior(f'layer{l}.{name}', value.shape)

    def fit(
        self, 
        data, 
        num_samples=1000, 
        num_steps=1000, 
        num_particles=10, 
        progress_bar=True, 
        opt_kwargs={'learning_rate': 1e-3}, 
        autoguide=None, 
        rank=2
        ):
        optimizer = optax_to_numpyro(self.optimizer(**opt_kwargs))
        model = self.model

        if autoguide == 'mean-field':
            guide = AutoNormal(self.model)
        elif autoguide == 'multivariate':
            guide = AutoMultivariateNormal(self.model)
        elif autoguide == 'lowrank-multivariate':
            guide = AutoLowRankMultivariateNormal(self.model, rank=rank)
        elif autoguide == 'bnaf-normal':
            guide = AutoBNAFNormal(self.model)
        elif autoguide == 'test':
            guide = self.guide
        else:
            guide = AutoDelta(self.model)
            
        loss = TraceGraph_ELBO(num_particles=num_particles)
        self.rng_key, _rng_key = random.split(self.rng_key)

        svi = SVI(model, guide, optimizer, loss)

        self.results = svi.run(_rng_key, num_steps, progress_bar=progress_bar, obs=data)
        
        pred = Predictive(model, guide=guide, params=self.results.params, num_samples=num_samples)
        
        self.samples = pred(_rng_key, obs=data)

        return self.samples

class BMRDNN(object):
    def __init__(self, rng_key, nnet, images, tau_0=1, sigma_0=1, subsample_size=64, rank=5):
        self.N, self.D = images.shape
        self.images = images

        self.rng_key = rng_key
        self.tau_0 = tau_0
        self.sigma_0 = sigma_0

        self.nnet = nnet

        params, static = eqx.partition(self.nnet, eqx.is_inexact_array)
        self.vals, self.aux = params.tree_flatten()

        self.subsample_size=subsample_size

        self.rank = rank

    def likelihood(self, nnet, labels):
        with plate("N", self.N, subsample_size=self.subsample_size):
            batch_x = subsample(self.images, event_dim=1)
            pred = nnet(batch_x)
            
            if labels is not None:
                batch_y = subsample(labels, event_dim=0)
            else:
                batch_y = None
            
            sample(
                "obs", dist.Categorical(logits=pred), obs=batch_y
            )

    def _register_network_vars(self):
        params, static = eqx.partition(self.nnet, eqx.is_inexact_array)
        vals, aux = params.tree_flatten()
        new_vals = []
        for l, layer in enumerate(vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    new_lv += (param(f'loc.layer{l}.{name}', value),)
                else:
                    new_lv += (value,)

            new_vals.append( layer.tree_unflatten(l_aux, new_lv) )

        vals = (new_vals,) + vals[1:]
        params = params.tree_unflatten(aux, vals)

        return eqx.combine(params, static)

    def _register_random_network_vars(self, nnet):
        params, static = eqx.partition(nnet, eqx.is_inexact_array)
        vals, aux = params.tree_flatten()
        new_vals = []
        for l, layer in enumerate(vals[0]):
            lv, l_aux = layer.tree_flatten()
            new_lv = ()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        new_lv += (sample(f'layer{l}.{name}', dist.Normal(value, self.sigma_0).to_event(1)),)
                    else:
                        new_lv += (sample(f'layer{l}.{name}', dist.Normal(value, self.sigma_0).to_event(1)),)
                else:
                    new_lv += (value,)

            new_vals.append( layer.tree_unflatten(l_aux, new_lv) )

        vals = (new_vals,) + self.vals[1:]
        params = params.tree_unflatten(aux, vals)

        return eqx.combine(params, static)

    def model(self, labels=None):
        d_nnet = self._register_network_vars()
        r_nnet = self._register_random_network_vars(d_nnet)
        Ws, _ = eqx.partition(r_nnet, eqx.is_inexact_array)
        lin_nnet = partial(linearize_nnet, d_nnet, Ws)
        self.likelihood(vmap(lin_nnet), labels)

    def guide(self, labels=None):
        rank = self.rank
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        name = f'layer{l}.{name}'
                        scale = param('scale.' + name, jnp.ones(value.shape)/10, constraint=dist.constraints.softplus_positive)
                        sample(name, dist.Normal(0., scale).to_event(1))
                    else:
                        name = f'layer{l}.{name}'
                        i, j = value.shape
                        scale_u = param('scale_u.' + name, jnp.ones((i, rank))/jnp.sqrt(10 * rank))
                        scale_v = param('scale_v.' + name, jnp.ones((rank, j))/jnp.sqrt(10 * rank))
                        scale = jnp.abs(scale_u @ scale_v)
                        sample(name, dist.Normal(0., scale).to_event(1))
        
    def aux_guide(self, labels=None):
        rank = self.rank
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'bias':
                        name = f'layer{l}.{name}'
                        scale = param('scale.' + name, jnp.ones(value.shape), constraint=dist.constraints.softplus_positive)
                        sample(name, dist.Normal(0., scale).to_event(1))
                    else:
                        name = f'layer{l}.{name}'
                        scale = param('scale.' + name, jnp.ones(value.shape))
                        sample(name, dist.Normal(0., scale).to_event(1))

    def gamma(self, c_sqr, v, u):
        v0 = jnp.expand_dims(v[..., 0], -1)
        u0 = jnp.expand_dims(u[..., 0], -1)
        return self.tau_0**2 * c_sqr * v[..., 1:] * v0/(c_sqr * u[..., 1:] * u0  + self.tau_0 * v[..., 1:] * v0)

    def ΔF(self, μ, σ_sqr, γ_sqr):
        π_0 = 1/self.sigma_0**2
        # π_{lij} = 1/σ^2
        π = 1/σ_sqr

        # m_{lij} = γ² + σ² - π₀σ²γ²

        m = jnp.clip(γ_sqr + σ_sqr - π_0 * σ_sqr * γ_sqr, a_min=1e-16)

        # \tilde{σ}^2_{lij} = γ² σ² / m
        _σ_sqr = γ_sqr * σ_sqr / m

        # \tilde{μ}_{lij} = π_{lij} μ_{lij} / \tilde{π}_{lij}
        _μ = π * μ * _σ_sqr
        
        df = - jnp.log(m).sum()
        df = df - jnp.sum(μ**2 * π * (1 - γ_sqr/m))

        return df/2, _μ, jnp.sqrt(_σ_sqr)
            
    def bmr_model(self, params):
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'weight':
                        name = f'layer{l}.{name}'
                        shape = value.shape
                        c_sqr = sample(f'{name}.c^2', dist.InverseGamma(2, 3))
                        with plate(name, shape[0]):
                            u = sample(f'{name}.u', dist.Gamma(1/2, 1).expand([shape[-1] + 1]).to_event(1))
                            v = sample(f'{name}.v', dist.Gamma(1/2, 1).expand([shape[-1] + 1]).to_event(1))

                            tau = deterministic(f'{name}.tau', self.tau_0 * jnp.sqrt(v[..., 0]/u[..., 0]))
                            lam = deterministic(f'{name}.lam', jnp.expand_dims(tau, -1) * jnp.sqrt(v[..., 1:]/u[..., 1:]))
                            γ_sqr = deterministic(f'{name}.gamma^2', self.gamma(c_sqr, v, u))
                            
                            μ = params['loc.' + name]
                            u = params['scale_u.' + name]
                            v = params['scale_v.' + name]
                            σ_sqr = jnp.square(u @ v)

                            log_prob, t_mu, t_scale = vmap(self.ΔF)(μ, σ_sqr, γ_sqr)
                            deterministic(f'loc.{name}', t_mu)
                            deterministic(f'scale.{name}', t_scale)
                            factor(f'{name}.log_prob', log_prob)

    def _log_normal(self, name, shape, scale=.1):
        loc = param(f'{name}.loc', jnp.zeros(tuple(shape)))
        scale = param(f'{name}.scale', scale*jnp.ones(tuple(shape)), constraint=dist.constraints.softplus_positive)
        return sample(name, dist.LogNormal(loc, scale).to_event(1))

    def bmr_guide(self, params):
        for l, layer in enumerate(self.vals[0]):
            lv, l_aux = layer.tree_flatten()
            for value, name in zip(lv, l_aux[0]):
                if value is not None:
                    if name == 'weight':
                        name = f'layer{l}.{name}'
                        shape = value.shape
                        loc_c = param(f'{name}.c^2.loc', jnp.zeros(1))
                        scale_c = param(f'{name}.c^2.scale', jnp.ones(1), constraint=dist.constraints.softplus_positive)
                        sample(f'{name}.c^2', dist.LogNormal(loc_c, scale_c))
                        with plate(name, shape[0]):
                            self._log_normal(f'{name}.u', [shape[0], shape[-1] + 1])
                            self._log_normal(f'{name}.v', [shape[0], shape[-1] + 1])

    def run_svi(self, rng_key, num_steps, model, guide, optimizer, loss, *args, **kwargs):
        svi = SVI(model, guide, optimizer, loss)
        return svi.run(rng_key, num_steps, *args, **kwargs)

    def fit(self, labels, num_steps=1000, num_particles=1, opt_kwargs={'learning_rate': 1e-3}):
        _model = self.model
        _guide = self.guide
        optimizer = optax_to_numpyro(optax.chain(optax.adabelief(**opt_kwargs)))

        self.rng_key, _rng_key = random.split(self.rng_key)
        result = self.run_svi(_rng_key, num_steps, _model, _guide, optimizer, Trace_ELBO(num_particles), labels=labels)

        self.losess = result.losses
        params = result.params

        return params

    def bmr(self, params, dev, num_samples=100, num_steps=1000, num_particles=1, opt_kwargs={'learning_rate': 1e-3}):
        _model = self.bmr_model
        _guide = self.bmr_guide
        optimizer = optax_to_numpyro(optax.chain(optax.adabelief(**opt_kwargs)))

        self.rng_key, _rng_key = random.split(self.rng_key)
        _rng_key = device_put(_rng_key, dev)
        params = device_put(params, dev)
        bmr_res = self.run_svi(_rng_key, num_steps, _model, _guide, optimizer, Trace_ELBO(num_particles), params)

        L = self.nnet.depth + 1
        return_sites = [f'layer{l}.weight.gamma^2' for l in range(L)] 
        return_sites += [f'loc.layer{l}.weight' for l in range(L)] 
        return_sites += [f'scale.layer{l}.weight' for l in range(L)]
        pred = Predictive(_model, guide=_guide, params=bmr_res.params, num_samples=num_samples, return_sites=return_sites)
        self.rng_key, _rng_key = random.split(self.rng_key)
        samples = pred(_rng_key, params)

        return bmr_res, samples

    def test_model(self, model, guide, params, images, labels, num_samples=1):
        pred = Predictive(model, guide=guide, params=params, num_samples=num_samples)
        n, _ = images.shape
        self.rng_key, _rng_key = random.split(self.rng_key)
        sample = pred(_rng_key, self.nnet, images, subsample_size=n)

        if num_samples > 1:
            acc = jnp.mean(sample['obs'] == labels, -1)
            print(acc.mean(), acc.std())
        else:
            print( jnp.mean(sample['obs'] == labels) )