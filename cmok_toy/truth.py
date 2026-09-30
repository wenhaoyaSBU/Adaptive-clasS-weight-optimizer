"""Analytic generation and physical laws, and reproducible stratified toys."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    ids: np.ndarray


class Truth:
    # Rows/classes: background, signal. Both covariance matrices are SPD.
    means = np.array([[-0.5, 0.4, -0.2, 0.2], [0.8, -0.3, 0.4, 0.0]])
    covariances = np.array([
        [
            [1.2, -.25, .15, 0], 
            [-.25, 1., 0, .2], 
            [.15, 0, 1.1, -.1], 
            [0, .2, -.1, .8]
        ],
        [
            [.9, .2, .1, 0], 
            [.2, 1.1, -.15, .1], 
            [.1, -.15, .8, .15], 
            [0, .1, .15, 1.2]
        ],
    ])
    coefficients = np.array([[-.15, .30, 0., -.10], [.35, -.20, .12, 0.]])

    def __init__(self, constant_weights=False):
        self.constant_weights = constant_weights
        self.log_z = np.array([self._log_normalizer(k) for k in range(2)])

    @staticmethod
    def forward(u):
        x = np.array(u, dtype=np.float64, copy=True)
        x[..., 2] += .6 * u[..., 0] * u[..., 1]
        x[..., 3] += .4 * u[..., 0] ** 2 - .3 * u[..., 1] ** 2
        return x

    @staticmethod
    def inverse(x):
        u = np.array(x, dtype=np.float64, copy=True)
        u[..., 2] -= .6 * x[..., 0] * x[..., 1]
        u[..., 3] -= .4 * x[..., 0] ** 2 - .3 * x[..., 1] ** 2
        return u

    def tilt(self, k):
        """log w(forward(u)) = a.u + (1/2) u.A.u."""
        a = self.coefficients[k]
        A = np.zeros((4, 4))
        A[0, 1] = A[1, 0] = .6 * a[2]
        A[0, 0], A[1, 1] = .8 * a[3], -.6 * a[3]
        return a, A

    def _log_normalizer(self, k):
        mu, cov = self.means[k], self.covariances[k]
        a, A = self.tilt(k)
        inv = np.linalg.inv(cov)
        precision = inv - A
        np.linalg.cholesky(precision)  # The Gaussian exponential moment must exist.
        h = inv @ mu + a
        return .5 * (h @ np.linalg.solve(precision, h) - mu @ inv @ mu
                     - np.linalg.slogdet(cov)[1] - np.linalg.slogdet(precision)[1])

    def log_mean_weights(self, x):
        if self.constant_weights:
            return np.broadcast_to(self.log_z, (len(x), 2)).copy()
        return np.asarray(x) @ self.coefficients.T

    def log_q(self, x):
        u = self.inverse(x)
        # The triangular nonlinear transformation has determinant one.
        result = []
        for mu, cov in zip(self.means, self.covariances):
            d = u - mu
            result.append(-.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1]
                                + np.einsum('ni,ij,nj->n', d, np.linalg.inv(cov), d)))
        return np.stack(result, axis=1)

    def log_lr(self, x, physical=True):
        log_densities = self.log_q(x)
        if physical:
            log_densities += self.log_mean_weights(x) - self.log_z
        return log_densities[:, 1] - log_densities[:, 0]

    def physical_log_prior_odds(self, rho):
        return float(np.log(rho[1] / rho[0]) + self.log_z[1] - self.log_z[0])

    def posterior(self, x, rho):
        return expit(self.log_lr(x) + self.physical_log_prior_odds(rho))

    def sample_x(self, rng, n, k, physical=False):
        mu, cov = self.means[k], self.covariances[k]
        if physical and not self.constant_weights:
            a, A = self.tilt(k)
            inv = np.linalg.inv(cov)
            cov = np.linalg.inv(inv - A)
            mu = cov @ (inv @ mu + a)
        return self.forward(rng.multivariate_normal(mu, cov, size=n))


def generate(truth, seed, n_background, n_signal, sigma_background=.4,
             sigma_signal=1., signed=False):
    """Independent generation/noise and 70/15/15 splits, preserving class counts."""
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in ('train', 'validation', 'test')}
    offset = 0
    for k, (n, sigma) in enumerate(zip((n_background, n_signal), (sigma_background, sigma_signal))):
        x = truth.sample_x(rng, n, k)
        mean = np.exp(truth.log_mean_weights(x)[:, k])
        noise = rng.normal(size=n)
        w = mean * (1 + sigma * noise if signed else np.exp(sigma * noise - .5 * sigma**2))
        indices = rng.permutation(n)
        groups = np.split(indices, [int(.7 * n), int(.85 * n)])
        for name, idx in zip(parts, groups):
            parts[name].append((x[idx], np.full(len(idx), k), w[idx], offset + idx))
        offset += n
    result = {}
    for name, blocks in parts.items():
        arrays = [np.concatenate([block[j] for block in blocks]) for j in range(4)]
        idx = rng.permutation(len(arrays[0]))
        result[name] = Sample(*(a[idx] for a in arrays))
    return result


def effective_size(w):
    return float(np.sum(w)**2 / np.sum(w**2))


def validate_truth(seed=123, n=200_000):
    truth = Truth()
    rng = np.random.default_rng(seed)
    checks = {}
    for k, name in enumerate(('background', 'signal')):
        x = truth.sample_x(rng, n, k)
        w = np.exp(truth.log_mean_weights(x)[:, k])
        z = float(np.exp(truth.log_z[k]))
        se = float(w.std(ddof=1) / np.sqrt(n))
        checks[name] = dict(analytic_z=z, mc_z=float(w.mean()), mc_se=se,
                            pull=float((w.mean() - z) / se))
    return checks
