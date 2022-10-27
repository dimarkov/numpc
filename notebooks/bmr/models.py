from signal import Sigmasks
import jax.numpy as jnp
import numpyro.distributions as dist
import optax
import equinox as eqx

from functools import partial
from jax import nn, lax, random, vmap, device_put
from numpyro import handlers, sample, plate, deterministic, factor, subsample, param
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform, AffineTransform, ComposeTransform, ExpTransform
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from numpyro.optim import optax_to_numpyro

def exact_blr(X, y, lam=1, mu_0=0., a_0=2., b_0=1.):
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
    def __init__(self, rng_key, X, p0=1, with_qr=False, reg_type='linear'):
        self.N, self.D = X.shape
        self.X = X
        self.rng_key = rng_key
        self.with_qr = with_qr
        self.p0 = p0
        self.type = reg_type # type of the rergression problem
        
        if self.with_qr:
            # use QR decomposition
            self.Q, self.R = jnp.linalg.qr(X)
            self.R_inv = jnp.linalg.inv(self.R)
        
    def model(self, obs=None):
        sigma_sqr = sample('sigma^2', dist.InverseGamma(2., 1.))
        sigma = deterministic('sigma', jnp.sqrt(sigma_sqr))
        
        tau0 = self.p0 / ((self.D - self.p0) * jnp.sqrt(self.N))
        tau = sample('tau', dist.HalfCauchy(1.))
            
        lam = sample('lam', dist.HalfCauchy(1.).expand([self.D]).to_event(1))

        gamma = jnp.sqrt(tau0**2 * tau**2 * lam **2 / (1 + tau0**2 * tau**2 * lam ** 2))
            
        if self.with_qr:
            rt = QRTransform(self.R, self.R_inv)
            aff = AffineTransform(0., sigma * gamma)
            ct = ComposeTransform([aff, rt])
            with handlers.reparam(config={"theta": TransformReparam()}):
                theta = sample(
                    'theta', 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.D]).to_event(1), ct)
                )

            deterministic('beta', rt.inv(theta))
            tmp = self.Q.dot(theta)

        else:
            aff = AffineTransform(0., sigma * gamma)
            with handlers.reparam(config={"beta": TransformReparam()}):
                beta = sample(
                    'beta', 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.D]).to_event(1), aff)
                )
            tmp = self.X.dot(beta)

        alpha = sample('alpha', dist.Normal(0., 10.))
        mu = deterministic('mu', alpha + tmp)
        
        with plate('data', self.N):
            if self.type == 'linear':
                sample('obs', dist.Normal(mu, sigma), obs=obs)
            elif self.type == 'logistic':
                sample('obs', dist.Bernoulli(logits=mu), obs=obs)
    
    def fit(self, data, num_samples=1000, warmup_steps=1000, num_chains=1, summary=False, progress_bar=True):
        self.rng_key, _rng_key = random.split(self.rng_key)

        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, 
                    num_warmup=warmup_steps, 
                    num_samples=num_samples, 
                    num_chains=num_chains,
                    chain_method='vectorized',
                    progress_bar=progress_bar)
        
        mcmc.run(_rng_key, obs=data)

        if summary:
            mcmc.print_summary()

        samples = mcmc.get_samples(group_by_chain=False)
        self.mcmc = mcmc
        self.samples = samples

        return samples


class BayesRegressionSVI(object):
    def __init__(self, rng_key, X, p0=1, with_qr=False, reg_type='linear'):
        self.N, self.D = X.shape
        self.X = X
        self.rng_key = rng_key
        self.with_qr = with_qr
        self.p0 = p0
        self.type = reg_type # type of the rergression problem
        
        if self.with_qr:
            # use QR decomposition
            self.Q, self.R = jnp.linalg.qr(X)
            self.R_inv = jnp.linalg.inv(self.R)
        
    def model(self, obs=None):
        sigma_sqr = sample('sigma^2', dist.InverseGamma(2., 1.))
        sigma = deterministic('sigma', jnp.sqrt(sigma_sqr))
        
        tau0 = self.p0 / ((self.D - self.p0) * jnp.sqrt(self.N))
        u = sample('u', dist.Gamma(1/2, 1).expand([self.D + 1]).to_event(1))
        v = sample('v', dist.Gamma(1/2, 1).expand([self.D + 1]).to_event(1))
        
        tau = deterministic('tau', tau0 * jnp.sqrt(v[0]/u[0]))
        lam = deterministic('lam', tau * jnp.sqrt(v[1:]/u[1:]))

        gamma = deterministic('gamma', jnp.sqrt(tau0**2 * v[1:] * v[0] / (u[1:] * u[0] + tau0**2 * v[1:] * v[0])))
            
        if self.with_qr:
            rt = QRTransform(self.R, self.R_inv)
            aff = AffineTransform(0., sigma * gamma)
            ct = ComposeTransform([aff, rt])
            with handlers.reparam(config={"theta": TransformReparam()}):
                theta = sample(
                    'theta', 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.D]).to_event(1), ct)
                )

            deterministic('beta', rt.inv(theta))
            tmp = self.Q.dot(theta)

        else:
            aff = AffineTransform(0., sigma * gamma)
            with handlers.reparam(config={"beta": TransformReparam()}):
                beta = sample(
                    'beta', 
                    dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.D]).to_event(1), aff)
                )
            tmp = self.X.dot(beta)

        alpha = sample('alpha', dist.Normal(0., 10.))
        mu = deterministic('mu', alpha + tmp)
        
        with plate('data', self.N):
            if self.type == 'linear':
                sample('obs', dist.Normal(mu, sigma), obs=obs)
            elif self.type == 'logistic':
                sample('obs', dist.Bernoulli(logits=mu), obs=obs)

    def fit(self, data, num_samples=1000, num_steps=1000, num_particles=1, opt_kwargs={'learning_rate': 1e-2}, autoguide=None):
        optimizer = optax_to_numpyro(optax.chain(optax.adabelief(**opt_kwargs)))
        model = self.model

        if autoguide == 'mean-field':
            guide = AutoNormal(self.model)
            loss = TraceMeanField_ELBO(num_particles=num_particles)
        elif autoguide == 'multivaraite':
            guide = AutoMultivariateNormal(self.model)
            loss = Trace_ELBO(num_particles=num_particles)
        elif autoguide == 'lowrank-multivaraite':
            guide = AutoLowRankMultivariateNormal(self.model, rank=2)
            loss = Trace_ELBO(num_particles=num_particles)
        else:
            guide = AutoDelta(self.model)
            loss = Trace_ELBO(num_particles=num_particles)

        self.rng_key, _rng_key = random.split(self.rng_key)

        svi = SVI(model, guide, optimizer, loss)

        self.results = svi.run(_rng_key, num_steps, obs=data)

        return_sites = ["beta", "sigma", "tau", "lam", "gamma"]
        pred = Predictive(model, guide=guide, params=self.results.params, num_samples=num_samples, return_sites=return_sites)
        self.samples = pred(_rng_key, obs=data)

        return self.samples, self.results.losses


class BMRNormalRegression(object):
    def __init__(self, rng_key, X, p0=1, with_qr=False, reg_type='linear'):
        self.N, self.D = X.shape
        self.X = X
        self.S = X.T @ X
        self.rng_key = rng_key
        self.with_qr = with_qr
        self.p0 = p0
        self.type = reg_type # type of the rergression problem
        self.tau_0 = p0 / ((self.D - p0) * jnp.sqrt(self.N))
        
        if self.with_qr:
            # use QR decomposition
            self.Q, self.R = jnp.linalg.qr(X)
            self.R_inv = jnp.linalg.inv(self.R)

    def gamma_2(self, v, u):
        return self.tau_0**2 * v[1:] * v[0] / (u[1:] * u[0] + self.tau_0**2 * v[1:] * v[0])

    def ΔF(self, a, b, mu, P,  S, g2):
        G = jnp.diag(g2)
        I = jnp.eye(self.D)

        _, logdet = jnp.linalg.slogdet(P)
        df = logdet/2
        
        R = G@S + I
        _, logdet = jnp.linalg.slogdet(R)
        df -= logdet/2
        
        Q = P @ jnp.linalg.solve(R, I - G)
        df -= a * jnp.log(1 + mu @ (Q @ mu)/(2*b))

        return df
            
    def model(self, a_n, b_n, mu_n, P_n, obs=None):

        u = sample('u', dist.Gamma(1/2, 1).expand([self.D + 1]).to_event(1))
        v = sample('v', dist.Gamma(1/2, 1).expand([self.D + 1]).to_event(1))
        
        tau = deterministic('tau', self.tau_0 * jnp.sqrt(v[0]/u[0]))
        lam = deterministic('lam', tau * jnp.sqrt(v[1:]/u[1:]))
        g2 = deterministic('gamma^2', self.gamma_2(v, u))

        log_prob = self.ΔF(a_n, b_n, mu_n, P_n,  self.S, g2)

        factor('log_prob', log_prob)

    def fit(self, data, num_samples=1000, num_steps=1000, num_particles=1, opt_kwargs={'learning_rate': 1e-2}, autoguide=None):

        mu_n, P_n, a_n, b_n = exact_blr(self.X, data)

        optimizer = optax_to_numpyro(optax.chain(optax.adabelief(**opt_kwargs)))
        model = self.model

        if autoguide == 'mean-field':
            guide = AutoNormal(self.model)
            loss = TraceMeanField_ELBO(num_particles=num_particles)
        elif autoguide == 'multivariate':
            guide = AutoMultivariateNormal(self.model)
            loss = Trace_ELBO(num_particles=num_particles)
        elif autoguide == 'lowrank-multivariate':
            guide = AutoLowRankMultivariateNormal(self.model, rank=2)
            loss = Trace_ELBO(num_particles=num_particles)
        else:
            guide = AutoDelta(self.model)
            loss = Trace_ELBO(num_particles=num_particles)

        self.rng_key, _rng_key = random.split(self.rng_key)

        svi = SVI(model, guide, optimizer, loss)

        self.results = svi.run(_rng_key, num_steps, a_n, b_n, mu_n, P_n, obs=data)

        return_sites = ["tau", "lam", "gamma^2"]
        pred = Predictive(model, guide=guide, params=self.results.params, num_samples=num_samples, return_sites=return_sites)
        self.samples = pred(_rng_key, a_n, b_n, mu_n, P_n, obs=data)

        blr = partial(exact_blr, self.X, data)
        mu_n, P_n, a_n, b_n = vmap(blr)(lam=1/self.samples['gamma^2'])

        self.rng_key, _rng_key = random.split(self.rng_key)
        sigma_square = dist.InverseGamma(a_n, b_n).sample(_rng_key)
        self.samples['sigma^2'] = sigma_square

        self.rng_key, _rng_key = random.split(self.rng_key)
        precision_matrix = jnp.expand_dims(1/sigma_square, (-1, -2)) * P_n
        beta = dist.MultivariateNormal(mu_n, precision_matrix=precision_matrix).sample(_rng_key)
        self.samples['beta'] = beta

        return self.samples, self.results.losses


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


class BMRDNN(object):
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

    def mv_guide(self, labels=None):
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
                        loc = param('mu.' + name, value)
                        i, j = value.shape
                        cov = param('cov.' + name, jnp.zeros((i, j, j)))
                        sample(name, dist.MultivariateNormal(loc, cov))

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
        v0 = jnp.expand_dims(v[..., 0], -1)
        u0 = jnp.expand_dims(u[..., 0], -1)
        return self.tau_0**2 * v[..., 1:] * v0 / (c_sqr * u[..., 1:] * u0 + self.tau_0**2 * v[..., 1:] * v0)

    def ΔF(self, mu, d, W, gamma):
        # Σ = D + W @ W.T
        Σ = jnp.diag(d) + W @ W.T
        # C = I + W.T @ inv(D) @ W
        C = jnp.eye(self.rank) + W.T @ (W/jnp.expand_dims(d, -1))
        # D' = G + D @ (I - G)
        _d = gamma + d * (1 - gamma)  
        # W' = (I - G) @ W
        _W = jnp.expand_dims(1 - gamma, -1) * W
        # M = D' + W @ W'.T
        M = jnp.diag(_d) + W @ _W.T
        # C' = I + W'.T @ inv(D') @ W
        _C = jnp.eye(self.rank) + _W.T @ ( W/jnp.expand_dims(_d, -1) )         
        
        # log|M| = log|D'| + log|C'|
        df = - jnp.log(_d).sum()/2

        _, logdet = jnp.linalg.slogdet(_C)
        df -= logdet/2

        # inv(M) = inv(D') - inv(D') @ W @ inv(C') @ W'.T @ inv(D') = inv(D') - R
        # Σ' = G @ inv(M) @ Σ = G @ inv(D') @ D - G @ R @ D + G @ inv(D') @ W @ W.T - G @ R @ W @ W.T
        _Σ = jnp.expand_dims(gamma, -1) * jnp.linalg.solve(M, Σ)

        # mu' = G @ inv(M) @ mu
        _μ = gamma * jnp.linalg.solve(M, mu)

        # inv(Σ) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D)
        tmp = _μ/d
        tmp = tmp - (W @ jnp.linalg.solve(C, W.T @ tmp))/d
        df += jnp.dot(mu, tmp)/2

        return df, _μ, _Σ
            
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
                            g = deterministic(f'{name}.gamma^2', self.gamma(c_sqr, v, u))
                            
                            mu = params['loc.' + name]
                            cov_diag = params['cov_diag.' + name]
                            cov_factor = params['cov_factor.' + name]

                            log_prob, t_mu, t_Sigma = vmap(self.ΔF)(mu, cov_diag, cov_factor, g)
                            deterministic(f'mu.{name}', t_mu)
                            deterministic(f'cov.{name}', t_Sigma)
                            factor(f'{name}.log_prob', log_prob)

    def _log_normal(self, name, shape, scale=.1):
        loc = param(f'{name}.loc', jnp.zeros(tuple(shape)))
        scale = param(f'{name}.scale', scale*jnp.ones(tuple(shape)), constraint=dist.constraints.softplus_positive)
        return sample(name, dist.LogNormal(loc, scale).to_event(1))

    def _log_lowrank_normal(self, name, shape, scale=.1):
        loc = param(f'{name}.loc', jnp.zeros(tuple(shape)))
        cov_diag = param(f'{name}.cov_diag', scale*jnp.ones(tuple(shape)), constraint=dist.constraints.softplus_positive)
        cov_factor = param('cov_factor.' + name, jnp.zeros(tuple(shape) + (self.rank,)))

        Exp = ExpTransform()
        mvn = dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        sample(name, dist.TransformedDistribution(mvn, Exp))

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
                            self._log_lowrank_normal(f'{name}.u', [shape[0], shape[-1] + 1])
                            self._log_lowrank_normal(f'{name}.v', [shape[0], shape[-1] + 1])

    def run_svi(self, rng_key, num_steps, model, guide, optimizer, loss, *args, **kwargs):
        svi = SVI(model, guide, optimizer, loss)
        return svi.run(rng_key, num_steps, *args, **kwargs)

    def fit(self, labels, num_steps=1000, num_particles=1, opt_kwargs={'learning_rate': 1e-3}):
        _model = self.model
        _guide = self.lowrank_guide
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
        return_sites = [f'mu.layer{l}.weight' for l in range(L)] 
        return_sites += [f'cov.layer{l}.weight' for l in range(L)]
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