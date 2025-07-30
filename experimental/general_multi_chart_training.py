"""
We want to make this the same as the applications/general_training.py

where you can switch between multi/single chart through a single config boolean

All the logic would be stored in applications/utils.perform_training.py
"""

import wandb

#get the relevant neural network classes to initialize phi,psi, g as
from core.template_psi_phi_g_functions_neural_networks import (
    identity_diffeomorphism,
    NN_diffeomorphism,
    NN_diffeomorphism_for_chart,
    NN_split_diffeomorphism,
    NN_linear_split_diffeomorphism,
    NN_Jacobian_split_diffeomorphism,
    NN_Jacobian_split_diffeomorphism_for_chart,
    NN_conv_diffeomorphism_for_chart,
    NN_conv_diffeomorphism_for_parametrization,
    identity_metric,
    NN_metric,
    NN_metric_regularized,
)

#get the relevant loss functions
from core.losses import (
    reconstruction_loss,
    input_target_loss,
    trajectory_reconstruction_loss,
    trajectory_prediction_loss,
    trajectory_loss
)

#get the relevant utility methods (HERE WE IMPORT THEM FROM EXPERIMENTAL INSTEAD OF APPLICATIONS)
from experimental.utils import (
    perform_training
)

#get the relevant methods to configure hyperparameters
from applications.configs import (
    get_wandb_config,
    PATH_LOGS
)


################################ initialize a Weights and Biases project ################################
wandb.init(project="Neural geodesic flows",
           group = "Geodesics on the sphere",
           dir=PATH_LOGS)

################################ choose all hyperparameters and neural networks used ################################

#get a wandb.config variable holding all hyper and high level parameters for the training run
#mandatory arguments: train/test_dataset_name, model_name, dim_dataspace, dim_M, psi/phi/g_arguments, batch_size

config = get_wandb_config(train_dataset_name  = "sphere_trajectories_train",
                          test_dataset_name = "sphere_trajectories_test",
                          model_name = "sphere_autoencoder",
                          dim_dataspace = 6,
                          dim_M = 2,
                          psi_arguments = {"in_size": 6,
                                           "out_size": 4,
                                           "hidden_sizes_x": [32, 32]},
                          phi_arguments = {"in_size": 4,
                                           "out_size": 6,
                                           "hidden_sizes_x": [32, 32]},
                          g_arguments = {'dim_M':2,
                                         'hidden_sizes':[32,32]},
                          batch_size = 512,
                          train_dataset_size = 16384,
                          test_dataset_size = 64,
                          learning_rate = 1e-3,
                          epochs = 100, loss_print_frequency = 25,
                          is_multi_chart = True,
                          continue_training = False,
                          updated_model_name = "",
                          save = True)

#choose the type of neural networks used for psi,phi, g. They have to have two arguments which is a dictionary,
#which also has to be saved as a member variable, and a random key.
#they will get passed the dictionary specified above in the get config variable.
#if doing continued training of a saved model, assign the initializers of the networks that the model previously used.
#if you forgot, the network class names are written in the model_name_high_level_params.json file (for this exact purpose)
psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric


#make sure that the chosen loss functions match the used datasets
#meaning they either take arguments & have keys
#inputs, targets, times
#or trajectories, times.
train_loss_function = trajectory_reconstruction_loss
test_loss_function = trajectory_reconstruction_loss


perform_training(config,
                 psi_initializer,
                 phi_initializer,
                 g_initializer,
                 train_loss_function,
                 test_loss_function)
