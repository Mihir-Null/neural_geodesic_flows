"""
Collection of auxiliarly methods adapted for the multi chart NGFs:

Once this here is in a satisfying state I will combine it into the applications/utils.py
"""

from chex import assert_equal
import jax
import jax.lax as lax
import jax.numpy as jnp

import optax
import equinox as eqx

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import wandb

import json

from core.models import (
    TangentBundle_single_chart_atlas
)

from experimental.atlas import (
    CoordinateDomain,
    create_coordinate_domains,
    create_atlas,
)

from experimental.models import (
    TangentBundle_multi_chart_atlas
)

#load variables with the relevant paths
from applications.configs import PATH_DATASETS
from applications.configs import PATH_MODELS

from applications.configs import (
    get_optimizer
)

from core.training import (
    train
)

#get the relevant inference methods
from core.inference import (
    input_target_model_analyis,
    trajectory_model_analyis,
    trajectory_model_visualization
)


#load a dataset stored in PATH_DATASETS/name.npz and return it as tuple of jax arrays.
#Optionally shrink the size by random selection or truncation.
#also return a string mode = 'input-target' or 'trajectory' to know what format the data has
def load_dataset(name, size=None, random_selection=False, key=jax.random.PRNGKey(0)):

    path = PATH_DATASETS/f"{name}.npz"

    loaded_data = np.load(path)

    keys = set(loaded_data.files)

    if {"inputs", "targets", "times"} <= keys:
        mode = "input-target"

        #convert to jax arrays.
        arrays = tuple(jnp.array(loaded_data[k]) for k in ["inputs", "targets", "times"])

    elif {"trajectories", "times"} <= keys:
        mode = "trajectory"

        #convert to jax arrays.
        arrays = tuple(jnp.array(loaded_data[k]) for k in ["trajectories", "times"])

    else:
        raise ValueError(f"Dataset {name} has unsupported keys: {loaded_data.files}")

    #perform the potential shrinking
    full_size = arrays[0].shape[0]

    #in case we don't want to shrink or specified as size bigger than the dataset, return the whole dataset (do nothing)
    if size is None or size >= full_size:

        print(f"\nLoaded full dataset {name} of type '{mode}' of size {full_size}\n")

    #else if we do want to shrink and passed a size smaller than the dataset, shrink it
    else:
        #shrink with random selection
        if random_selection:
            indices = jax.random.choice(key, full_size, (size,), replace=False)
        #or with truncation
        else:
            indices = jnp.arange(size)

        #actual shrinking
        arrays = tuple(arr[indices, ...] for arr in arrays)

        print(f"\nLoaded dataset {name} of type '{mode}' of shrunk size {size}.\n")

    return arrays, mode

#THIS METHOD IS PAINFULLY SLOW. REWRITE! (in c++?)
#expect to be given a tuple of arrays (trajectories, times) of shapes
#(many, timepoints, mathim) and (many, timepoints)
#and mebmerships of shape
#(many, timepoints, amount of domains)
#saying for each point if it belongs to the i-th domain (i-th component = 1) or not (i-th component = 0).
#We then take the global trajectories and create domain specific constructed_trajectories for each domain,
#which are padded with nans and constructed_times which are fully populated and start at 0.
#We return a tuple of tuples, (constructed_trajectories, constructed_times) for each chart.
def create_domain_specific_data_arrays(dataset_arrays, memberships):
    all_trajectories, all_times = dataset_arrays
    N, T, D = all_trajectories.shape
    K = memberships.shape[-1]

    def extract_segments(trajectory):
        max_segs = T // 2
        segments = jnp.full((max_segs, T, D), jnp.nan)

        i = 0
        seg_id = 0

        while i < T - 1:
            seg_traj = []

            k = 0
            while i + k < T and not jnp.isnan(trajectory[i + k, 0]):
                seg_traj.append(trajectory[i + k])
                k += 1

            if seg_traj:
                seg_arr = jnp.array(seg_traj)
                segments = segments.at[seg_id, 0:k, :].set(seg_arr)
                seg_id += 1

            i += k + 1

        return segments[:seg_id]  # only return filled segments

    domain_specific_arrays = ()

    for k in range(K):
        domain_mask = memberships[..., k].astype(bool)[..., None]  # (N, T, 1)
        domain_mask = jnp.broadcast_to(domain_mask, all_trajectories.shape)
        masked_trajectories = jnp.where(domain_mask, all_trajectories, jnp.nan)

        all_segments = []

        for i in range(N):
            segs = extract_segments(masked_trajectories[i])
            all_segments.append(segs)

        # flatten into (num_segs_total, T, D)
        constructed_trajectories = jnp.concatenate(all_segments, axis=0)

        # mask out invalid segments (e.g., fully NaN) — optional safety check
        valid_mask = ~jnp.all(jnp.isnan(constructed_trajectories), axis=(1, 2))
        constructed_trajectories = constructed_trajectories[valid_mask]

        # create time arrays (starting from 0)
        time = all_times[0]
        constructed_times = jnp.tile(time, (constructed_trajectories.shape[0], 1))

        domain_specific_arrays += ((constructed_trajectories, constructed_times),)

    return domain_specific_arrays




#create a dataloader for a dataset given as a tuple arrays = (array_0, ..., array_k)
#expect axis 0 to be the batch dimension, i.e. array_i has shape (many, ...)
def create_dataloader(dataset_arrays, batch_size):

    #convert jax arrays to pytorch tensors
    tensors = tuple(torch.tensor(np.array(arr)) for arr in dataset_arrays)

    #account for potential case batch_size == None (test dataloaders might have this)
    if batch_size is None:
        batch_size = tensors[0].shape[0]

    #create a tensor dataset and a torch DataLoader
    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nCreated DataLoader with batch size {batch_size}\n")

    return dataloader


#load a model of the type TangentBundle stored in PATH_MODELS/model_name.eqx and return it.
#Here we are using the models high level parameters stored
#in PATH_MODELS/model_name_high_level_params.json in order to load it correctly.
#You have to pass the initializers (just pass the classes) of psi, phi and g.
#that were used in training. If you forgot, their names are written in the json file (for this purpose).
def load_model(model_name, psi_initializer, phi_initializer, g_initializer):

    #paths of saved model and high level parameters
    model_path = PATH_MODELS/f"{model_name}.eqx"
    model_high_level_params_path = PATH_MODELS/f"{model_name}_high_level_params.json"

    #load the model high level parameters from the json file
    with open(model_high_level_params_path, 'r') as f:
        model_high_level_params = json.load(f)

    is_multi_chart = model_high_level_params['is_multi_chart']        #needs to become model parameter

    if not is_multi_chart:

        #initialize correct type neural networks (could also be hardcoded function, dependinging on the initializer)
        psi_NN = psi_initializer(model_high_level_params['psi_arguments'])

        phi_NN = phi_initializer(model_high_level_params['phi_arguments'])

        g_NN = g_initializer(model_high_level_params['g_arguments'])

        #using the models high level parameters create an instance of the exact same form
        model_prototype = TangentBundle_single_chart_atlas(dim_dataspace = model_high_level_params['dim_dataspace'],
                                                           dim_M = model_high_level_params['dim_M'],
                                                           psi = psi_NN, phi = phi_NN, g = g_NN)

    elif is_multi_chart:

        domains = ()

        for domain_shape in model_high_level_params['domain_shapes']:

            centroid = jnp.zeros(domain_shape['centroid_shape'])
            interior_points = jnp.zeros(domain_shape['interior_points_shape'])
            boundary_points = jnp.zeros(domain_shape['boundary_points_shape'])
            boundary_new_chart_ids = jnp.zeros(domain_shape['boundary_new_chart_ids_shape'], dtype=jnp.int32)

            domain = CoordinateDomain(centroid, interior_points, boundary_points, boundary_new_chart_ids)

            domains = domains + (domain,)


        atlas = create_atlas(domains = domains,
                             psi_initializer = psi_initializer,
                             phi_initializer = phi_initializer,
                             g_initializer = g_initializer,
                             psi_arguments = model_high_level_params['psi_arguments'],
                             phi_arguments = model_high_level_params['phi_arguments'],
                             g_arguments = model_high_level_params['g_arguments'],
                             key = jax.random.PRNGKey(0))

        model_prototype = TangentBundle_multi_chart_atlas(atlas = atlas)

    #initialize the saved model
    model = eqx.tree_deserialise_leaves(model_path, like = model_prototype)

    print(f"\nLoaded model {model_name}\n")

    return model


#Takes a trained model of type eqx.module containg a method get_high_level_parameters (which returns a dictionary)
#and stores it and its high level parameters
#under PATH_MODELS/model_name.eqx and PATH_MODELS/model_name_high_level_params.json
def save_model(model, model_name):

    #obtain model high level parameters (such as dim_M ...)
    model_high_level_params = model.get_high_level_parameters()

    #store the trained model locally in Models
    eqx.tree_serialise_leaves(PATH_MODELS/f"{model_name}.eqx", model)

    #store the model parameters locally in Models
    with open(PATH_MODELS/f"{model_name}_high_level_params.json", 'w') as f:
        json.dump(model_high_level_params, f)

    print(f"\nSaved model under the name {model_name}\n")


#perform the training of a model with the loading and saving methods defined above
#and with weight and biases for hyperparameter management.
#return the model.
def perform_training(config,
                    psi_initializer,
                    phi_initializer,
                    g_initializer,
                    train_loss_function,
                    test_loss_function):


    #update the run name
    if config.continue_training:
        wandb.run.name = config.updated_model_name
        print(f"\nContinuing run {config.model_name} as {config.updated_model_name}")
    else:
        wandb.run.name = config.model_name
        print(f"\nCommencing run {config.model_name}")

    #initialize top level random key, all others will be splits of this
    top_level_key = jax.random.PRNGKey(config.random_seed)


    ################################ load the problems dataset ################################
    key, key_train_loader, key_test_loader = jax.random.split(top_level_key, 3)

    train_dataset_arrays, train_dataset_mode = load_dataset(name = config.train_dataset_name,
                                                            size = config.train_dataset_size,
                                                            random_selection = True,
                                                            key = key_train_loader)

    test_dataset_arrays, test_dataset_mode = load_dataset(name = config.test_dataset_name,
                                                          size = config.test_dataset_size,
                                                          random_selection = True,
                                                          key = key_test_loader)

    
    assert_equal(train_dataset_mode, test_dataset_mode)



    ################################ single chart case ################################
    if not config.is_multi_chart:

       
        ########################### load or initialize a model #########################
        if config.continue_training:

            model = load_model(config.model_name, 
                               psi_NN_initializer = psi_initializer,
                               phi_NN_initializer = phi_initializer,
                               g_NN_initializer = g_initializer)
        
        else:

            key, key_psi, key_phi, key_g = jax.random.split(key, 4)

            psi_NN = psi_initializer(config.psi_arguments, key = key_psi)

            phi_NN = phi_initializer(config.phi_arguments, key = key_phi)

            g_NN = g_initializer(config.g_arguments, key = key_g)

            model = TangentBundle_single_chart_atlas(dim_dataspace = config.dim_dataspace, dim_M = config.dim_M,
                                                     psi = psi_NN, phi = phi_NN, g = g_NN)

            
        ########################### create the data loaders ############################
        
        train_dataloader = create_dataloader(dataset_arrays = train_dataset_arrays,
                                             batch_size = config.batch_size)

        test_dataloader = create_dataloader(dataset_arrays = test_dataset_arrays,
                                            batch_size = config.test_dataset_size)


        ########################### perform the training ###############################

        optimizer = get_optimizer(name = config.optimizer_name, learning_rate = config.learning_rate)

        #train the model
        model = train(model = model,
                      train_loss_function = train_loss_function,
                      test_loss_function = test_loss_function,
                      train_dataloader = train_dataloader,
                      test_dataloader = test_dataloader,
                      optimizer = optimizer,
                      epochs = config.epochs,
                      loss_print_frequency = config.loss_print_frequency)


        


    ################################ multi chart case ##################################
    else:

        ########################### prepare by obtaining domains and memberships #######
        if not train_dataset_mode == 'trajectory':

            raise ValueError("Multi chart training is only supported for data of mode 'trajectory'")
        
        #assume thus that the data or mode 'trajectory', extract trajectories while ignoring the times
        train_trajectories, _ = train_dataset_arrays
        test_trajectories, _ = test_dataset_arrays

        #obtain an array with all points from all trajectories
        train_datamanifold = train_trajectories.reshape(-1, train_trajectories.shape[-1])  # shape (many*time points, math dim)
        test_datamanifold = test_trajectories.reshape(-1, test_trajectories.shape[-1])  # shape (many*time points, math dim)


        #we split the data manifold into domains, which is a tuple of elements of type CoordinateDomain
        #memberships are shape (many,amount of domains) where many = datamanifold.shape[0] = amount of trajectories * timepoints
        train_domains, train_memberships = create_coordinate_domains(train_datamanifold,
                                                                     amount_of_domains = 2,      #promote to hyper parameter!
                                                                     extension_degree = 0,       #promote to hyper parameter!
                                                                     is_tangent_bundle = True)   #promote to hyper parameter!

        test_domains, test_memberships = create_coordinate_domains(test_datamanifold,
                                                                   amount_of_domains = 2,
                                                                   extension_degree = 0,
                                                                   is_tangent_bundle = True)


        #reshape the memberships to (many, timepoints, amount of charts), so that it works with trajectories of shape (many, timepoints, mathdim)
        train_memberships = train_memberships.reshape(train_trajectories.shape[0], train_trajectories.shape[1], train_memberships.shape[1])
        test_memberships = test_memberships.reshape(test_trajectories.shape[0], test_trajectories.shape[1], test_memberships.shape[1])


        ########################### load or initialize a model #########################
        if config.continue_training:

            model = load_model(config.model_name,
                               psi_NN_initializer = psi_initializer,
                               phi_NN_initializer = phi_initializer,
                               g_NN_initializer = g_initializer)
       
        else:
       
            key, key_atlas = jax.random.split(key, 2)

            #we assign an instance of Chart to each of the domains. All charts have the same psi,phi,g architecture.
            #this yields a tuple of Charts called atlas.
            atlas = create_atlas(domains = train_domains,
                                 psi_initializer = psi_initializer,
                                 phi_initializer = phi_initializer,
                                 g_initializer = g_initializer,
                                 psi_arguments = config.psi_arguments,
                                 phi_arguments = config.phi_arguments,
                                 g_arguments = config.g_arguments,
                                 key = key_atlas)

            model = TangentBundle_multi_chart_atlas(atlas = atlas)


        ########################### create the data loaders ############################
        
        #create a tuple of dataset arrays, one for each domain of the model, like so
        #tuple = ( (trajectories_1, times_1), ..., (trajectories_k, times_k)
        train_domain_specific_dataset_arrays = create_domain_specific_data_arrays(train_dataset_arrays, train_memberships)
        test_domain_specific_dataset_arrays = create_domain_specific_data_arrays(test_dataset_arrays, test_memberships)

        #create a tuple of train dataloaders
        train_dataloaders = ()

        for dataset_arrays in train_domain_specific_dataset_arrays:

            train_dataloader = create_dataloader(dataset_arrays = dataset_arrays,
                                                 batch_size = config.batch_size)

            train_dataloaders += (train_dataloader,)

        #create a tuple of test dataloaders
        test_dataloaders = ()

        for dataset_arrays in test_domain_specific_dataset_arrays:

            test_dataloader = create_dataloader(dataset_arrays = dataset_arrays,
                                                batch_size = config.test_dataset_size)

            test_dataloaders += (test_dataloader,)


        ########################### perform the training ###############################
        trained_atlas = ()

        for i, chart in enumerate(model.atlas):

            print(f"\n### Now training chart {i} of {len(model.atlas)-1} ###\n")

            optimizer = get_optimizer(name = config.optimizer_name, learning_rate = config.learning_rate)

            #train the model
            trained_chart = train(model = chart,
                                  train_loss_function = train_loss_function,
                                  test_loss_function = test_loss_function,
                                  train_dataloader = train_dataloaders[i],
                                  test_dataloader = test_dataloaders[i],
                                  optimizer = optimizer,
                                  epochs = config.epochs,
                                  loss_print_frequency = config.loss_print_frequency)

            trained_atlas += (trained_chart,)

        model = TangentBundle_multi_chart_atlas(atlas = trained_atlas)


    ################################ save the model ################################
    if config.save:
        if config.continue_training:

            save_model(model, config.updated_model_name)
            wandb.finish()
            print(f"\nFinished training model {config.updated_model_name}.")

        else:
            save_model(model, config.model_name)
            wandb.finish()
            print(f"\nFinished training model {config.model_name}.")

    return model


#perform inference of a trained model with the loading methods defined above
#and inference methods defined in core/inference.py
def perform_inference(model_name,
                      psi_initializer,
                      phi_initializer,
                      g_initializer,
                      dataset_name,
                      dataset_size,
                      seed = 0):

    #load the model
    model = load_model(model_name,
                       psi_initializer = psi_initializer,
                       phi_initializer = phi_initializer,
                       g_initializer = g_initializer)

    #load the data, respecting the correct mode
    data, mode = load_dataset(name = dataset_name,
                 size=dataset_size,
                 random_selection=True,
                 key=jax.random.PRNGKey(seed))

    if mode == "input-target":
        inputs, targets, times = data

    elif mode =="trajectory":
        trajectories, times = data


    #perform the analysis respecting the correct mode
    if mode == "input-target":
        input_target_model_analyis(model, inputs, targets, times)

    elif mode =="trajectory":
        trajectory_model_analyis(model, trajectories, times)

        #trajectory_model_visualization(model, trajectories, times)
