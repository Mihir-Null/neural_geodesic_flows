"""
Training script for pseudo-Riemannian NGF on Schwarzschild radial trajectories.
"""

import wandb

from core.template_psi_phi_g_functions_neural_networks import (
    NN_Jacobian_split_diffeomorphism,
    NN_pseudo_metric_fixed_signature,
)

from core.losses import trajectory_loss

from applications.utils import perform_training
from applications.configs import get_wandb_config, PATH_LOGS


################################ initialize a Weights and Biases project ################################
wandb.init(project="Neural geodesic flows",
           group="Schwarzschild radial",
           dir=PATH_LOGS)

################################ choose all hyperparameters and neural networks used ################################

config = get_wandb_config(
    train_dataset_name="schwarzschild_radial_train",
    test_dataset_name="schwarzschild_radial_test",
    model_name="schwarzschild_prngf",
    dim_dataspace=4,
    dim_M=2,
    psi_arguments={"in_size": 4, "out_size": 4, "hidden_sizes_x": [64, 64]},
    phi_arguments={"in_size": 4, "out_size": 4, "hidden_sizes_x": [64, 64]},
    g_arguments={
        "dim_M": 2,
        "hidden_sizes": [64, 64],
        "signature": (1, 1),
        "min_diagonal": 1e-2,
        "min_scale": 1e-2,
    },
    batch_size=128,
    train_dataset_size=None,
    test_dataset_size=None,
    learning_rate=5e-4,
    epochs=50,
    loss_print_frequency=5,
    continue_training=False,
    updated_model_name="",
    save=True,
)

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_pseudo_metric_fixed_signature


def prngf_loss(model, trajectories, times):
    # add degeneracy penalty (stronger conditioning)
    return trajectory_loss(
        model,
        trajectories,
        times,
        metric_reg_weight=1e-2,
        metric_logabsdet_floor=-12.0,
    )


train_loss_function = prngf_loss
test_loss_function = prngf_loss

perform_training(
    config,
    psi_initializer,
    phi_initializer,
    g_initializer,
    train_loss_function,
    test_loss_function,
)
