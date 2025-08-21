"""
This module contains functions and classes for multi-object observations used for inference in linear Gaussian state space models (LGSSMs).
"""
import jax.numpy as jnp
import jax.random as jr

import jax.debug
from jax import vmap
from jax import lax
from jaxtyping import Array, Float
from dynamax.utils.utils import psd_solve, symmetrize
from typing import NamedTuple, Optional, Union, Tuple

from abc import ABC, abstractmethod

def MVN_multiply(m1: Array, c1: Array, m2: Array, c2: Array) -> Tuple[float, Tuple[Array, Array]]:
    '''
    Calculates the product of gaussian densities

    Source: the matrix cookbook eqn 371


    Parameter
    ---------
    m1: Array
        mean of first gaussian
    c1: Array
        covariance of first gaussian
    m2: Array
        mean of second gaussian
    c2: Array
        covariance of second gaussian

    Returns
    -------
    tuple[float, tuple[Array, Array]]
        float, product_gaussian

    '''

    k = m1.shape[-1]
    mean_diff = m1 - m2
    cov = c1 + c2
    # c = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.inv(cov) @ mean_diff)
    c = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.solve(cov, mean_diff))

    s_cov_inv = jnp.linalg.inv(c1)
    o_cov_inv = jnp.linalg.inv(c2)

    cov = jnp.linalg.inv(s_cov_inv+o_cov_inv)
    mean = cov @ (s_cov_inv @ c1.T + o_cov_inv @ c2.T)

    return c, (mean, cov)

def MVN_inverse_bayes(prior: Tuple[Array, Array], posterior: Tuple[Array, Array]):
    '''
    Determines the gaussian likelihood function given a gaussian posterior and prior

    Derivation is simple using natural parameters

    Parameter
    ---------
    prior: tuple[Array, Array]
        gaussian prior function
    posterior: tuple[Array, Array]
        gaussian posterior function
    
    Returns
    -------
    tuple[Array, Array]
        gaussian likelihood function
    '''
    # if \Sigma_posterior^-1 - \Sigma_prior^-1 is not positive semidefinite then this is an invalid covariance matrix
    issue = jnp.any(jnp.isnan(jnp.linalg.cholesky(jnp.linalg.inv(posterior[1]) - jnp.linalg.inv(prior[1]))))

    # \Sigma_l = (\Sigma_posterior^-1 - \Sigma_prior^-1)^-1
    likelihood_sigma = jnp.linalg.inv(jnp.linalg.inv(posterior[1]) - jnp.linalg.inv(prior[1]))
    # \mu_l = \Sigma_l @ (\Sigma_posterior^-1 @ \mu_posterior - \Sigma_prior^-1 @ \mu_prior)
    likelihood_mu = likelihood_sigma @ (jnp.linalg.inv(posterior[1]) @ posterior[0] - jnp.linalg.inv(prior[1]) @ prior[0])
    
    return (likelihood_mu, likelihood_sigma)

def GMM_moment_match(dists: Tuple[Array, Array], weights: Array) -> Tuple[Array, Array]:
    ''' Finds a gaussian with moments matching a multivariate distribution

    Test cases:
    ```python
    from dynamax.utils.plotting import plot_uncertainty_ellipses
    dists = (jnp.array([[-3], [3]]), jnp.array([[2], [2]]))
    weights = jnp.array([0.5, 0.5])

    # should be ([0], [11])
    print(GMM_moment_match(dists, weights))


    dist1 = (jnp.array([1, 0]), jnp.array([[1, 0], [0, 1]])*0.1)
    dist2 = (jnp.array([-1, 0]), jnp.array([[1, 0], [0, 1]])*0.1)
    weight = jnp.array([0.5, 0.5])

    dist1 = (jnp.array([1, 1]), jnp.array([[1, 0], [0, 1]])*0.1)
    dist2 = (jnp.array([-1, -1]), jnp.array([[1, 0], [0, 1]])*0.1)
    weight = jnp.array([0.5, 0.5])

    dist1 = (jnp.array([1, 1]), jnp.array([[1, 0], [0, 1]])*0.1)
    dist2 = (jnp.array([-1, -1]), jnp.array([[1, 0], [0, 1]])*0.1)
    weight = jnp.array([0.95, 0.05])

    combined = GMM_moment_match((jnp.array([dist1[0], dist2[0]]), jnp.array([dist1[1], dist2[1]])), weight)

    fig, ax = plt.subplots()
    # plot_uncertainty_ellipses from dynamax.utils.plotting
    plot_uncertainty_ellipses([dist1[0], dist2[0]], [dist1[1], dist2[1]], ax)
    plot_uncertainty_ellipses([combined[0]], [combined[1]], ax, edgecolor='tab:red')

    ax.set_xlim(-5,6)
    ax.set_ylim(-5,6)
    plt.show()
    ```
    '''
    # \mu = \Sum w_{k,i} mu_i
    mu = weights @ dists[0]

    # \Sigma = \Sum w_{k,i} * (\Sigma_i + (\mu - \mu_i) @ (\mu - \mu_i)^T)
    mean_diff = mu - dists[0]
    sigma = jnp.average(dists[1] + vmap(jnp.outer)(mean_diff, mean_diff), weights=weights, axis=0)

    return (mu, sigma)

class Emission(ABC):
    @abstractmethod
    def emit(self, t: int, pred_mean, pred_cov, F, B, b, Q, H, D, d, u) -> Tuple[Float[Array, "emission_dim"], Float[Array, "emission_dim emission_dim"]]:
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass

class EmissionDynamicCovariance(Emission):
    def __init__(self, emissions: Float[Array, "num_timesteps emission_dim"], emission_covs: Float[Array, "num_timesteps emission_dim emission_dim"]):
        self.emissions = emissions
        self.emission_covs = emission_covs
    
    def emit(self, t: int, pred_mean, pred_cov, F, B, b, Q, H, D, d, u) -> Tuple[Float[Array, "emission_dim"], Float[Array, "emission_dim emission_dim"]]:
        return self.emissions[t], self.emission_covs[t]

    def __len__(self) -> int:
        return self.emissions.shape[0]

class EmissionConstantCovariance(EmissionDynamicCovariance):
    def __init__(self, emissions: Array, R: Array):
        super().__init__(emissions, jnp.broadcast_to(R, (emissions.shape[0], emissions.shape[1], emissions.shape[1])))

class EmissionPDA(Emission):
    def __init__(self, emissions: Float[Array, "num_timesteps num_observations emission_dim"], emission_covs: Float[Array, "num_timesteps num_observations emission_dim emission_dim"], condition_on, sharpening=1.0):
        self.emissions = emissions
        self.emission_covs = emission_covs
        self.sharpening = sharpening

        self.condition_on = condition_on

        self.combined_emissions = jnp.zeros((emissions.shape[0], emissions.shape[2]))
        self.combined_emission_covs = jnp.zeros((emissions.shape[0], emissions.shape[2], emissions.shape[2]))

    def evaluate_observation(self, emission_mean, emission_cov, pred_mean, pred_cov, H, D, d, u):
        filtered_mean, filtered_cov = self.condition_on(pred_mean, pred_cov, H, D, d, emission_cov, u, emission_mean)

        filtered_emission_mean = H @ filtered_mean
        filtered_emission_cov = H @ filtered_cov @ H.T

        w_k = MVN_multiply(filtered_emission_mean, filtered_emission_cov, emission_mean, emission_cov)[0]

        return (filtered_mean, filtered_cov), w_k
    
    def emit(self, t: int, pred_mean, pred_cov, F, B, b, Q, H, D, d, u) -> Tuple[Float[Array, "emission_dim"], Float[Array, "emission_dim emission_dim"]]:
        # find GMM that best represents observations
        (filtered_means, filtered_covs), w_ks = vmap(
            lambda emission_mean, emission_cov: self.evaluate_observation(
                emission_mean, emission_cov,
                pred_mean, pred_cov,
                H, D, d, u
            )
        )(self.emissions[t], self.emission_covs[t])

        # normalize list to fix underflow issues
        w_ks = w_ks - jnp.max(w_ks) # jax.lax.stop_gradient(jnp.max(w_ks))
        
        # sharpen
        w_ks = w_ks * self.sharpening
        # move out of log space
        w_ks = jnp.exp(w_ks)
        # normalize
        w_ks = w_ks / w_ks.sum()
        
        filtered_mean, filtered_cov = GMM_moment_match((filtered_means, filtered_covs), w_ks)
        
        filtered_emission_mean = H @ filtered_mean
        filtered_emission_cov = H @ filtered_cov @ H.T
        pred_emission_mean = H @ pred_mean
        pred_emission_cov = H @ pred_cov @ H.T
        emission_mean, emission_cov = MVN_inverse_bayes((pred_emission_mean, pred_emission_cov), (filtered_emission_mean, filtered_emission_cov))

        return emission_mean, emission_cov

    def __len__(self) -> int:
        return self.emissions.shape[0]
