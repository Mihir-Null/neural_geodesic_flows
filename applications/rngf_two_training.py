"""
Training script for Riemannian (SPD) NGF baseline on two-body trajectories (thesis dataset).
"""

import wandb

from core.template_psi_phi_g_functions_neural_networks import (
    NN_Jacobian_split_diffeomorphism,
    NN_metric_regularized,
)

from core.losses import trajectory_loss

from applications.utils import perform_training
from applications.configs import get_wandb_config, PATH_LOGS


################################ initialize a Weights and Biases project ################################
wandb.init(project="Neural geodesic flows",
           group="Two-body trajectories",
           dir=PATH_LOGS)

################################ choose all hyperparameters and neural networks used ################################

config = get_wandb_config(
    train_dataset_name="two-body-problem_trajectories_train",
    test_dataset_name="two-body-problem_trajectories_test",
    model_name="two_body_rngf",
    dim_dataspace=8,
    dim_M=4,
    psi_arguments={"in_size": 8, "out_size": 8, "hidden_sizes_x": [64, 64]},
    phi_arguments={"in_size": 8, "out_size": 8, "hidden_sizes_x": [64, 64]},
    g_arguments={
        "dim_M": 4,
        "hidden_sizes": [64, 64],
    },
    batch_size=128,
    train_dataset_size=None,
    test_dataset_size=None,
    learning_rate=5e-4,
    epochs=10,
    loss_print_frequency=5,
    continue_training=False,
    updated_model_name="",
    save=True,
)

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric_regularized


def rngf_loss(model, trajectories, times):
    return trajectory_loss(
        model,
        trajectories,
        times,
        metric_reg_weight=0.0,
    )


train_loss_function = rngf_loss
test_loss_function = rngf_loss

perform_training(
    config,
    psi_initializer,
    phi_initializer,
    g_initializer,
    train_loss_function,
    test_loss_function,
)
