"""
Inference methods for multi-chart NGFs. These are currently a bit ad-hoc to visualize what we've implemented so far.
"""

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

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

from core.inference import (
    apply_model_function,
    find_indices
)

#return x,y,z for plotting a 2d surface in 3d. Pass a parametrization function
def parametrized_surface(parametrization, chartdomain, grid_res = (30,30)):

    #embed the manifold into 3d
    x0_min, x0_max, x1_min, x1_max, x0_name, x1_name = chartdomain()

    #create grid
    x0_vals = jnp.linspace(x0_min, x0_max, grid_res[0])
    x1_vals = jnp.linspace(x1_min, x1_max, grid_res[1])
    grid = jnp.array(jnp.meshgrid(x0_vals, x1_vals))

    #flatten the grid to shape (res0*res1, 2)
    grid_flat = grid.reshape(2, -1).T

    #map the grid points into the data space shape (res0*res1, 3)
    mapped_grid = jax.vmap(parametrization, in_axes=0)(grid_flat)

    #reshape mapped grid back to match grid structure (res0, res1, 3)
    mapped_grid = mapped_grid.reshape(grid_res[0], grid_res[1], -1)

    #extract x, y, z for plotting
    x_grid, y_grid, z_grid = mapped_grid[..., 0], mapped_grid[..., 1], mapped_grid[..., 2]

    surface = (x_grid, y_grid, z_grid)

    return surface

def full_dynamics_visualization(tangent_bundle,
                                 initial_state,
                                 t,
                                 steps,
                                 surface):  # surface = (x_grid, y_grid, z_grid)


    #encode the initial_state
    initial_latent = tangent_bundle.psi(initial_state)

    #integration, in the chart, yielding a tuple (chart_ids, z_values)
    chart_geodesic = tangent_bundle.exp_return_trajectory(initial_latent, t, steps)

    #embed the integrated curve into data space, this is now an array of shape (steps+1, 6) (x,y,z,vx,vy,vz)
    geodesic = jax.vmap(tangent_bundle.phi, in_axes = 0)(chart_geodesic)

    #find the surface x,y,z for plotting
    x_grid, y_grid, z_grid = surface


    #prepare chart arrays
    chart_ids, zs = chart_geodesic

    #prepare colors
    amount_of_charts = tangent_bundle.amount_of_charts
    cmap = cm.get_cmap("winter", amount_of_charts)
    chart_colors = [cmap(i) for i in range(amount_of_charts)]


    #plot setup with the surface
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.5, hspace=0.6)
    #gs = gridspec.GridSpec(3, 4, figure=fig)  # 3 rows × 4 columns

    # Data space plot spans the top two columns
    ax_main = fig.add_subplot(gs[0:2, 1:3], projection='3d')
    #ax_main = fig.add_subplot(2, 3, 2, projection='3d')  # center top

    x_grid, y_grid, z_grid = surface
    ax_main.plot_wireframe(x_grid, y_grid, z_grid, color="gray", alpha=0.3)

    #plot geodesic in the data space (colors as they will be in the charts)
    for i in range(len(geodesic) - 1):
        cid = int(chart_ids[i])
        seg = geodesic[i:i + 2]
        ax_main.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=chart_colors[cid], linewidth=2)

    #mark initial point and initial tangent vector
    ax_main.scatter(*geodesic[0, :3], color='black', marker='o', label='initial')
    ax_main.quiver(*geodesic[0, :3], *geodesic[0, 3:], color='black', length=0.25)

    ax_main.set_xlabel(r'$y^1$')
    ax_main.set_ylabel(r'$y^2$')
    ax_main.set_zlabel(r'$y^3$')
    ax_main.set_title("Dynamics in data space")
    ax_main.axis('equal')
    ax_main.legend()

    #plot all charts, each with a different color
    rows = (amount_of_charts + 2) // 3
    plot_idx = 4  # subplot index for chart visualizations

    for i in range(amount_of_charts):

        mask = chart_ids == i

        z_i = zs[mask]

        ax = fig.add_subplot(2, 3, plot_idx)

        ax.plot(z_i[:, 0], z_i[:, 1], '.', color=chart_colors[i])

        ax.set_title(f'chart {i}')
        ax.set_xlabel(r'$x^1$')
        ax.set_ylabel(r'$x^2$')
        ax.axis('equal')
        plot_idx += 1

    #plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()



#this method visualizes data trajectories vs predictions of any NGF model automatically,
#regardless of single/multi chart and data/latent dimensions.
#you need to specify if the data space is itself a tangent bundle or not
def trajectory_model_visualization(model, trajectories, times, data_are_tangent_bundle, type = 'average', size = 10):

    ### in this first section we simply generate the predictions ###

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

        

    ### in this second section we determine which type of model and data we are dealing with ###
   
    #latent space is tangent bundle TM, so dim M is half the dimension
    
    #in the multi chart case latent states are tuples (chart_id, z)
    if model.is_multi_chart:
        dim_M = encoded_trajectories[1].shape[-1]//2

    #in the single chart case latent states are points z
    else:
        dim_M = encoded_trajectories.shape[-1]//2
   
    #data space dimension
    dim_dataspace = trajectories.shape[-1]

    #the data/latent may be high dimensional, which is why we will visualize them as (several) particles,
    #or just (several) 1d dimensions (if neither divisible by 2 or 3)

    latent_3d = False
    latent_2d = False
    
    #either we have some 3d particles
    if dim_M % 3 == 0:
        latent_amount_particles = dim_M//3
        latent_3d = True
    
    #or some 2d particles
    elif dim_M % 2 == 0:
        latent_amount_particles = dim_M//2
        latent_2d = True
    
    #or a collection of 1d lines
    else:
        latent_amount_particles = 0


    #some story with the data
    data_3d = False
    data_2d = False

    #either the data live in a tangent bundle    
    if data_are_tangent_bundle:

        half_dim_dataspace = dim_dataspace // 2
        
        #and we have some 3d particles
        if half_dim_dataspace % 3 == 0:
            data_amount_particles = half_dim_dataspace // 3
            data_3d = True

        #or some 2d particles
        elif half_dim_dataspace % 2 == 0:
            data_amount_particles = half_dim_dataspace // 2
            data_2d = True
            
        #or a collection of 1d lines
        else:
            data_amount_particles = 0
            data_lines = True

    #or the data are not on a tangent bundle
    else:

        #and we have some 3d particles
        if dim_dataspace % 3 == 0:
            data_amount_particles = dim_dataspace // 3
            data_3d = True

        #or some 2d particles
        elif dim_dataspace % 2 == 0:
            data_amount_particles = dim_dataspace // 2
            data_2d = True

        #or a collection of 1d lines
        else:
            data_amount_particles = 0



    ### in this third section we construct the subplots ###

    """ We put the data plot top left,
        and the k charts in a grid next to it, like so
        # # # #
          # # #
        (example was for six charts), or like so
        # #
        (example for a single chart).
        To get the exact grid layout we will take the root of k,
        round it down, and make this the amount of rows.
        Then we fill it up with as many columns as need.
        The last column may possibly only be partially populated.

        The subplot should either be 2d (2d particles or lines) or 3d (3d particles)
    """ 
    #multi chart case, rows and columns of the chart grid (only!)
    if model.is_multi_chart:

        rows = int(jnp.floor(jnp.sqrt(model.amount_of_charts)))

        columns = int(jnp.ceil(model.amount_of_charts / rows)) 
  
        amount_of_charts = model.amount_of_charts

    #single chart case, rows and columns of the chart grid (only!)
    else:
        rows = 1
        columns = 1
   
        amount_of_charts = 1


    fig = plt.figure(figsize=(24, 12))

    #to add subplots do:
    #  ax = fig.add_subplot(nrows, ncols, index, projection='3d') (last part optional, only for 3d)
    #the subplots will be on a grid with nrows rows and ncols columns.
    #ax will be at index, where 1 is in the upper left corner and it increases to the right first and down second.


    #data plot, add one column for the data plot
    
    #(several) 3d particles
    if data_3d:

        ax = fig.add_subplot(rows, 1+columns, 1, projection='3d')
    
        #populate it

        #if tangent bundle, quiver, else scatter


    #(several) 2d particles
    elif data_2d:

        ax = fig.add_subplot(rows, 1+columns, 1)

        #populate it

        #if tangent bundle, quiver, else scatter

    #(several) lines
    else:

        ax = fig.add_subplot(rows, 1+columns, 1)




    #latent chart plots
    for column in range(columns):

        for row in range(rows):

            #this is just within the chart grid, filling it up columnwise
            index = 1 + row*columns + column 
            
            if index <= amount_of_charts:

                #(several) 3d particles
                if latent_3d:
                                                          #add a correction because of the data column
                    ax = fig.add_subplot(rows, 1+columns, 1 + row + index, projection='3d')

                    ### populate it ###



                #(several) 2d particles
                elif latent_2d:
                                                          #add a correction because of the data column
                    ax = fig.add_subplot(rows, 1+columns, 1 + row + index) 
                
                ### populate the chart plot ###


                #lines
                else:
                                                          #add a correction because of the data column
                    ax = fig.add_subplot(rows, 1+columns, 1 + row + index) 
                
                ### populate the chart plot ###




    #adjust layout and display the plot
    plt.tight_layout()
    plt.show()

