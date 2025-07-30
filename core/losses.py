"""
Collection of loss functions
"""

import jax
import jax.numpy as jnp

#the below prediction losses needs to behave differently in the single and multi chart approach,
#namely, in the multi chart approach latent points are tuples (chart_id, z), in the single chart they are arrays z
#we check which one it is at trace time using this auxiliary method
#(so that it does not re-check every call, but just once at trace time)
def is_multi_chart(latent):

    return isinstance(latent, tuple) and len(latent) == 2

#in the multi chart training the data trajectories are padded with NaNs.
#this means square error method filters out NaN paddings and stops their gradients
def safe_filtered_MSE(data, predictions):
    
    #we will filter out and stop gradients of any NaNs in the data (suppossedly NaN paddings)
    valid_mask = ~jnp.isnan(data)
    
    count_valid = jnp.sum(valid_mask)
    
    #find the difference
    diff = data - predictions
    
    #stop gradients in NaN paddings
    diff = jnp.where(valid_mask, diff, jax.lax.stop_gradient(diff))
    
    #build the square error filtering out the NaN paddings
    square_error = jnp.where(valid_mask, diff**2, 0.0)
    
    #find the MSE
    mse = jnp.sum(square_error) / count_valid   

    return mse

#this method applies the function to a point safely:
#if it's non nan, it just applies it
#if it's nan, it returns the point itself (so nan) (so that it can later be filtered out since evidently it belongs to the padding area)
#but stops the gradient, so that the model call on a nan value won't ruin the gradients (it otherwise would)
def safe_function_apply(function, point):

    is_valid = ~jnp.isnan(point).any()

    def do_model(point):       
            
        return function(point)

    def skip_model(point):

        #returns the point unchanged, and no gradients!
        return jax.lax.stop_gradient(function(point))

    output = jax.lax.cond(is_valid, do_model, skip_model, point)
        
    return output

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

    #measure the MSE of the predicted versus the target in latent space
    if is_multi_chart(latent_predictions):

        _, z_pred = latent_predictions
        _, z_targ = latent_targets

        latent_predictive_error = jnp.mean((z_pred - z_targ)**2)

    else:

        latent_predictive_error = jnp.mean((latent_predictions - latent_targets)**2)

    #measure the quality of the reconstruction by MSE
    reconstructive_error = jnp.mean((reconstructions_inputs - inputs)**2 + (reconstructions_targets - targets)**2)

    #loss as a weighted combination (as they have the same units the weight should be in [0,1])
    return reconstructive_error + predictive_error + latent_predictive_error

#expect data of shape (batch_size, time steps, mathematical dimension), (batch_size,time steps)
def trajectory_reconstruction_loss(tangentbundle, trajectories, times):

    #build and vectorize the functions from the autoencoder
    autoencode = lambda point : tangentbundle.phi(tangentbundle.psi(point))
    safe_autoencode = lambda point : safe_function_apply(autoencode, point)

    autoencode_points = jax.vmap(safe_autoencode, in_axes = 0)

    #reshape to points (batch_size*time steps, math dim)
    points = trajectories.reshape(-1, trajectories.shape[-1])

    #generate reconstructions
    reconstructions = autoencode_points(points)

    reconstructive_error = safe_filtered_MSE(points, reconstructions)

    #loss
    return reconstructive_error


#expect data of shape (batch_size, time steps, mathematical dimension), (batch_size,time steps)
def trajectory_prediction_loss(tangentbundle, trajectories, times):

    #find the final times and the number of steps to take (assume times are equidistant)
    final_times = times[:,-1]
    num_steps = times.shape[1] - 1 #time steps = num_steps + 1 (the first time step will be the initial and for each num step we go forward by one step)


    #vectorize the encoder
    safe_encode = lambda point : safe_function_apply(tangentbundle.psi, point)

    #expect to be given a trajectory (num_steps + 1, math dim)
    encode_trajectory = jax.vmap(safe_encode, in_axes = 0)
    #expect to be given a batch of trajectories (many, num steps + 1, math dim)
    encode_many_trajectories = jax.vmap(encode_trajectory, in_axes = 0)

    

    #vectorize the decoder. Will only be applied to generated geodesic (so no nans)
    
    #expect to be given a geodesic (num steps + 1, math dim)
    decode_geodesic = jax.vmap(tangentbundle.phi, in_axes = 0)
    #expect to be given a batch of geodesics (many, num steps +1 , math dim)
    decode_many_geodesics = jax.vmap(decode_geodesic, in_axes = 0)

    #vectorize the geodesic solve

    #expect to be given a batch of initial points (many, math dim). Initial points must never be NaN
    encode_initial = jax.vmap(tangentbundle.psi, in_axes = 0)

    #expect to be given a batch of encoded initial points (many, math dim) as well as final_times (many,)
    find_geodesic = jax.vmap(tangentbundle.exp_return_trajectory, in_axes = (0,0,None))




    #generate the predicted trajectories (which are geodesics) and match them
    #with the given trajectories in both data and latent space.

    #encode the given trajectories
    encoded_trajectories = encode_many_trajectories(trajectories)

    #encode all the initial points from the given trajectories, get shape (many, math dim)
    encoded_initial_points = encode_initial(trajectories[:,0,...])

    #find all the corresponding geodesics, get shape (many, num steps +1, math dim)
    geodesics = find_geodesic(encoded_initial_points, final_times, num_steps)
    
    #decode all the geodesics, get shape (many, num_steps + 1, math dim)
    decoded_geodesics = decode_many_geodesics(geodesics)
    
    
    #measure the deviation of the predicted versus the given trajectory in latent space
    if is_multi_chart(geodesics):
        _, z_geodesics = geodesics
        _, z_encoded_trajectories = encoded_trajectories

        predictive_error_latentspace = safe_filtered_MSE(z_encoded_trajectories, z_geodesics)

    else:

        predictive_error_latentspace = safe_filtered_MSE(encoded_trajectories, geodesics)
    
    #measure the deviation of the predicted versus the given trajectory in dataspace
    predictive_error_dataspace = safe_filtered_MSE(trajectories, decoded_geodesics)

    return predictive_error_dataspace + predictive_error_latentspace


def trajectory_loss(tangentbundle, trajectories, times):

        #find the predictive error (will be latent + dataspace)
        predictive_error = trajectory_prediction_loss(tangentbundle, trajectories, times)

        #find the reconstructive error
        reconstructive_error = trajectory_reconstruction_loss(tangentbundle, trajectories, times)

        #return the sum of prediction and reconstruction error
        return predictive_error + reconstructive_error
