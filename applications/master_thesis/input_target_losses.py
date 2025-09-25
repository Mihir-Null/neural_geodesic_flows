"""
Collection of input target loss functions (rather than trajectory loss).
We did some tests with those in the master thesis.
In general using trajectories is way better, which is why we only include trajectory losses
in the more module core/losses.py

This module contains the losses as well as their unit tests.
Run this module as main to execute the tests.
"""

import jax
import jax.numpy as jnp

from tests.utils import (
    printheading,
    test_function_dimensionality,
)

from core.models import (
    TangentBundle_single_chart_atlas as TangentBundle,
)

from core.template_psi_phi_g_functions_analytical import (
    psi_S2_normal,
    phi_S2_normal,
    g_S2_normal,
)

#expect data of shape (batch_size, mathematical dimension), (batch_size,mathematical dimension), (batch_size)
def reconstruction_loss(tangentbundle, inputs, targets, times):

    #vectorize the functions from the tangentbundle
    encoder = jax.vmap(tangentbundle.psi, in_axes = 0)
    decoder = jax.vmap(tangentbundle.phi, in_axes = 0)

    #generate reconstructions
    reconstructions_inputs = decoder(encoder(inputs))
    reconstructions_targets = decoder(encoder(targets))

    #measure the quality of the reconstruction by MSE
    reconstructive_power = jnp.mean((reconstructions_inputs - inputs)**2) + jnp.mean((reconstructions_targets - targets)**2)

    #loss
    return reconstructive_power

#expect data of shape (batch_size, mathematical dimension), (batch_size,mathematical dimension), (batch_size)
def input_target_loss(tangentbundle, inputs, targets, times):

    #vectorize the functions from the tangentbundle
    exp = jax.vmap(tangentbundle.exp, in_axes = (0,0,None))
    encoder = jax.vmap(tangentbundle.psi, in_axes = 0)
    decoder = jax.vmap(tangentbundle.phi, in_axes = 0)

    #generate predictions
    num_steps = 49

    latent_inputs = encoder(inputs)
    latent_targets = encoder(targets)

    latent_predictions = exp(latent_inputs, times, num_steps)

    predictions = decoder(latent_predictions)

    #generate reconstructions
    reconstructions_inputs = decoder(latent_inputs)
    reconstructions_targets = decoder(latent_targets)

    #measure the quality of the predicition by MSE
    predictive_error = jnp.mean((predictions - targets)**2)

    latent_predictive_error = jnp.mean((latent_predictions - latent_targets)**2)

    #measure the quality of the reconstruction by MSE
    reconstructive_error = jnp.mean((reconstructions_inputs - inputs)**2 + (reconstructions_targets - targets)**2)

    #loss as a weighted combination (as they have the same units the weight should be in [0,1])
    return reconstructive_error + predictive_error + latent_predictive_error


def unit_test_recon_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda inputs, targets, times : reconstruction_loss(tangentbundle = tangentbundle,
                                                            inputs =inputs, targets = targets,
                                                                times = times)

    printheading(unit_name="reconstruction_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,6),(100,6),(100,)])

def unit_test_input_target_loss():

    tangentbundle = TangentBundle(dim_dataspace = 6, dim_M = 2,
                                    psi = psi_S2_normal, phi = phi_S2_normal,
                                        g = g_S2_normal)

    loss = lambda inputs, targets, times : input_target_loss(tangentbundle = tangentbundle,
                                                            inputs =inputs, targets = targets,
                                                                times = times)

    printheading(unit_name="input_target_loss")

    test_function_dimensionality(func = loss, in_shapes = [(100,6),(100,6),(100,)])

#if this module is executed as main, do the unit tests
if __name__ == "__main__":
    unit_test_recon_loss()
    unit_test_input_target_loss()
