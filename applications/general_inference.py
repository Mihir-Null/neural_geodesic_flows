"""
Run model analysis (inference) methods.

Load a trained model and then run analysis functions,
passing a testdataset.

The specifics are meant to be adapted to run your different experiments.
"""

#get the relevant neural network classes to initialize psi,phi, g as
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

from applications.utils import (
    perform_inference
)

#define a test dataset (has to be one saved in data/datasets/)
dataset_name = "half-sphere_trajectories_test"
dataset_size = 1024

#define a saved model (has to be one saved in data/models/)
model_name = "half-sphere_model"

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric

#above assign the initializers of psi, phi and g that the model used,
#their names are written in the model_name_high_level_params.json file

#analyse the chosen model on the chosen test data
perform_inference(model_name,
                  psi_initializer,
                  phi_initializer,
                  g_initializer,
                  dataset_name,
                  dataset_size)
