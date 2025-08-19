"""
This module contains functions and classes for multi-object observations used for inference in linear Gaussian state space models (LGSSMs).
"""
import jax.numpy as jnp
import jax.random as jr

from jax import lax
from jaxtyping import Array, Float
from dynamax.utils.utils import psd_solve, symmetrize
from typing import NamedTuple, Optional, Union, Tuple

from abc import ABC, abstractmethod

class Emission(ABC):
    @abstractmethod
    def emit(self, t: int, F, B, b, Q, H, D, d) -> Tuple[Float[Array, "emission_dim"], Float[Array, "emission_dim emission_dim"]]:
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def means(self) -> Float[Array, "num_timesteps emission_dim"]:
        pass

    @abstractmethod
    def covs(self) -> Float[Array, "num_timesteps emission_dim emission_dim"]:
        pass

class EmissionDynamicCovariance(Emission):
    def __init__(self, emissions: Float[Array, "num_timesteps emission_dim"], emission_covs: Float[Array, "num_timesteps emission_dim emission_dim"]):
        self.emissions = emissions
        self.emission_covs = emission_covs
    
    def emit(self, t: int, F, B, b, Q, H, D, d) -> Tuple[Float[Array, "emission_dim"], Float[Array, "emission_dim emission_dim"]]:
        return self.emissions[t], self.emission_covs[t]

    def __len__(self) -> int:
        return self.emissions.shape[0]
    
    def means(self) -> Float[Array, "num_timesteps emission_dim"]:
        return self.emissions

    def covs(self) -> Float[Array, "num_timesteps emission_dim emission_dim"]:
        return self.emission_covs

class EmissionConstantCovariance(EmissionDynamicCovariance):
    def __init__(self, emissions: Array, R: Array):
        super().__init__(emissions, jnp.broadcast_to(R, (emissions.shape[0], emissions.shape[1], emissions.shape[1])))