"""
Definition of several model inference methods.

The essential methods are apply_model_function and find_indices.

All others call them, and use their return values to generate statistics or visualizations.

We use two different formats of the data:
- trajectories & times,       expected to be of shape (many, trajectory points, mathdim) & (many, trajectory points)
- points,                     expected to be of shape (many, mathdim)

There is only one visual methods currently implemented.
It'd be good to generalize it, so that it works out of the box for multi/single chart,
dataspaces that are themselves tanget bundles or not, and
any data and latent dimensions.
"""

import jax
import jax.numpy as jnp

import numpy as np

#set a backend for interactive plotting
import matplotlib
matplotlib.use('QtAgg')

#customize the figure default style and format
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "axes3d.mouserotationstyle": "azel",
})

import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull

#apply a method of the model to given input data, return the models outputs
#data is supposed to be a tuple of jax arrays, exactly as many as the arguments of model_function
def apply_model_function(model_function, data, vmap_axes):

    #vmap the function along the axis 0 for each input array in the data tuple
    function = jax.vmap(model_function, in_axes = vmap_axes)

    #apply the function with each array in the tuple data as an input
    outputs = function(*data)

    return outputs

#find the indices of the 10 (or size many) worst, average and best performing cases
def find_indices(correct_outputs, model_outputs, size = 10):

    #compute squared error across all but the "many" axis (axis 0)
    error_axes = tuple(range(1, correct_outputs.ndim))
    errors = jnp.mean((correct_outputs - model_outputs) ** 2, axis=error_axes)

    #find the size many indices with the smallest, average, and largest predictive error.
    sorted_indices = jnp.argsort(errors)

    best_indices = sorted_indices[:size]
    worst_indices = sorted_indices[-size:]
    avg_start = len(sorted_indices) // 2 - size // 2
    average_indices = sorted_indices[avg_start : avg_start + size]

    return worst_indices, average_indices, best_indices


#perform reconstruction and prediction analysis on trajectory style data
def trajectory_model_analyis(model, trajectories, times):

    #perform reconstruction on the whole trajectories
    autoencode = lambda y : model.phi(model.psi(y))
    autoencode_traj = jax.vmap(autoencode, in_axes = 0) #this will vmap along one trajectory

    recon = apply_model_function(autoencode_traj,
                                 tuple((trajectories,)),
                                 vmap_axes = (0)) #this will vmap along all trajectories

    #perform prediction starting from the initial points on the trajectories until the final times with time-points - 1 steps.
    pred = apply_model_function(model.get_geodesic,
                                tuple((trajectories[:,0,:], times[:,-1], times.shape[1] - 1)),
                                vmap_axes = (0,0,None))

    #find the reconstruction and prediction errors
    recon_error = jnp.mean((trajectories - recon)**2)
    pred_error = jnp.mean((trajectories - pred)**2)

    print(f"Reconstruction mean square error {recon_error}\n")
    print(f"Prediction mean square error {pred_error}\n")


#THIS METHOD IS HARDCODED FOR DATA SPACES THAT ARE THEMSELVES TANGENT BUNDLES
#We show given data space trajectories and a models prediction of said trajectories
#in the data and latent space. Trajectories have shape (amount of trajectoies, amount of time points, math dim)
#We group them into particles, if dim/2 = 2*d then into d-many 2D particles
#                              if dim/2 = 3*d ten into d-many 3D particles
#                              if dim/2 = 3*d +2 or +4 then into d-many 3D plus one or two 2D particles (this edge case is not implemented)
def trajectory_model_visualization(model, trajectories, times, type = 'average', size = 10):

    #first determine the trajectories in the latent space
    encode_trajectory = jax.vmap(model.psi, in_axes = 0)
    encode_trajectories = jax.vmap(encode_trajectory, in_axes = 0)

    encoded_trajectories = encode_trajectories(trajectories)

    #second generate and encode the predictions
    predictions = apply_model_function(model.get_geodesic,
                                       tuple((trajectories[:,0,:],
                                              times[:,-1],
                                              times.shape[1] - 1)),
                                        vmap_axes = (0,0,None))

    encoded_predictions = encode_trajectories(predictions)

    #third, decide what type of predictions to show
    worst_indices, average_indices, best_indices = find_indices(correct_outputs = trajectories,
                                                                model_outputs = predictions,
                                                                size = size)

    if type == 'worst':
        indices = worst_indices
        print(f"Visualizing the {size} worst trajectory predictions")

    elif type == 'average':
        indices = average_indices
        print(f"Visualizing {size} average trajectory predictions")

    elif type == 'best':
        indices = best_indices
        print(f"Visualizing the {size} best trajectory predictions")

    else:
        raise ValueError("type has to be 'worst', 'average' or 'best'")

    #find data and latent space dimensions
    half_dim_dataspace = trajectories.shape[-1]//2 #unfortunately this method assumes that the data space is a tangent bundle
    dim_M = encoded_trajectories.shape[-1]//2 #latent space is tangent bundle TM

    #find out what combination of 3*d, 2*d the data, latent space are
    if half_dim_dataspace % 2 == 0 and half_dim_dataspace % 2 == 0:

        print("Visualizers for even dim data and even dim latent not implemented")



    elif half_dim_dataspace % 2 == 0 and half_dim_dataspace % 3 == 0:

        print("Visualizers for even dim data and divisble by 3 dim latent not implemented")



    elif half_dim_dataspace % 3 == 0 and dim_M % 2 == 0:

        particles_data = half_dim_dataspace//3
        particles_latent = dim_M//2

        colors_data_traj = cm.get_cmap("Greys", particles_data + 1)
        colors_data_pred = cm.get_cmap("YlOrRd", particles_data + 1)

        colors_latent_traj = cm.get_cmap("Greys", particles_latent + 1)
        colors_latent_pred = cm.get_cmap("YlOrRd", particles_latent + 1)

        #plotting
        fig = plt.figure(figsize=(24, 12))

        #left: data trajectories and predictions
        ax1 = fig.add_subplot(121, projection='3d')

        for part in range(1, particles_data + 1):
            color_traj = colors_data_traj(part)
            color_pred = colors_data_pred(part)

            #plot the data trajectories with initial point in black
            ax1.scatter(trajectories[indices,0,0*part], trajectories[indices,0,1*part], trajectories[indices,0,2*part], color=color_traj, marker = 'o', s = 20, label=f'data particle {part}')
            ax1.quiver(trajectories[indices,0,0*part], trajectories[indices,0,1*part], trajectories[indices,0,2*part],
                       trajectories[indices,0,3*part], trajectories[indices,0,4*part], trajectories[indices,0,5*part],
                       color=color_traj, length = 0.25)

            for i in indices:
                ax1.plot(trajectories[i, :, 0*part], trajectories[i, :, 1*part], trajectories[i, :, 2*part], color=color_traj)

            #plot the data trajectories with initial point in red
            ax1.scatter(predictions[indices,0,0*part], predictions[indices,0,1*part], predictions[indices,0,2*part], color=color_pred, marker = 'o', s = 20, label=f'predictions particle {part}')
            ax1.quiver(predictions[indices,0,0*part], predictions[indices,0,1*part], predictions[indices,0,2*part],
                       predictions[indices,0,3*part], predictions[indices,0,4*part], predictions[indices,0,5*part],
                       color=color_pred, length = 0.25)

            for i in indices:
                ax1.plot(predictions[i, :, 0*part], predictions[i, :, 1*part], predictions[i, :, 2*part], color=color_pred)

        ax1.set_xlabel(r'$y^1$')
        ax1.set_ylabel(r'$y^2$')
        ax1.set_zlabel(r'$y^3$')
        ax1.set_title(r'flow in the data space $\tilde{N}$')
        ax1.legend()
        ax1.axis('equal')

        #right: chart trajectories / predictions
        ax2 = fig.add_subplot(122)

        for part in range(1, particles_latent + 1):
            color_traj = colors_latent_traj(part)
            color_pred = colors_latent_pred(part)

            #plot the charted data trajectories with intial point in red
            ax2.scatter(encoded_trajectories[indices,0,0*part], encoded_trajectories[indices,0,1*part],
                        color=color_traj, marker = 'o', s = 20, label=f'data particle {part}')

            ax2.quiver(encoded_trajectories[indices,0,0*part], encoded_trajectories[indices,0,1*part],
                       encoded_trajectories[indices,0,2*part], encoded_trajectories[indices,0,3*part],
                       color=color_traj, scale=7, scale_units="xy", width=0.003)
            for i in indices:
                ax2.plot(encoded_trajectories[i, :, 0*part], encoded_trajectories[i, :, 1*part], color=color_traj)

            #plot the charted data trajectories with intial point in red
            ax2.scatter(encoded_predictions[indices,0,0*part], encoded_predictions[indices,0,1*part],
                        color=color_pred, marker = 'o', s = 20, label=f'predictions particle {part}')

            ax2.quiver(encoded_predictions[indices,0,0*part], encoded_predictions[indices,0,1*part],
                       encoded_predictions[indices,0,2*part], encoded_predictions[indices,0,3*part],
                       color=color_pred, scale=7, scale_units="xy", width=0.003)
            for i in indices:
                ax2.plot(encoded_predictions[i, :, 0*part], encoded_predictions[i, :, 1*part], color=color_pred)

        ax2.set_xlabel(r'$x^1$')
        ax2.set_ylabel(r'$x^2$')
        ax2.set_title(r'flow in the chart of the latent space $N$')
        ax2.legend()
        ax2.axis('equal')

        #adjust layout and display the plot
        plt.tight_layout()
        plt.show()


    elif half_dim_dataspace % 3 == 0 and dim_M % 3 == 0:

        print("Visualizers for divisble by 3 dim data and divisble by 3 dim latent not implemented")
