"""
Model inference for a multi chart model.
First we do the standard numerical inference with the methods from core.inference (would work for single chart too)

Second some custom visual inference, which uses ad-hoc methods from experimental.inference
The goal would be to standardize those and have them become part of the standard program in core.inference
"""

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_S2_spherical,
    parametrization_S2_spherical,
)

from core.template_psi_phi_g_functions_neural_networks import (
    identity_metric,
    NN_Jacobian_split_diffeomorphism,
    NN_metric
)

from experimental.utils import (
    load_dataset,
    load_model,
)

from experimental.inference import (
    parametrized_surface,
    full_dynamics_visualization
)

from experimental.utils import (
    perform_inference
)


### define some test data ###
dataset_name = "sphere_trajectories_train"
size = 512

### define a saved model (has to be one saved in data/models/) ###
model_name = "sphere_ngf"

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric

### standard numerical inference ###
perform_inference(model_name=model_name, psi_initializer=psi_initializer,
                  phi_initializer=phi_initializer,
                  g_initializer=g_initializer,
                  dataset_name=dataset_name,
                  dataset_size=size)





### custom visual inference ###

### for a nice plot ### 
embedding = parametrized_surface(parametrization_S2_spherical, chartdomain_S2_spherical)


### load the test data ###
data, _ = load_dataset(name = dataset_name, size = size)

trajectories, times = data

### load the model ###
tangent_bundle = load_model(model_name,
                           psi_initializer = psi_initializer,
                           phi_initializer = phi_initializer,
                           g_initializer = g_initializer)

#initial point
initial_point = trajectories[200,0,:]

#integration time
t = 10

#integration steps
steps = 250

#visualize the geodesic in data space and in the charts
embedding = parametrized_surface(parametrization_S2_spherical, chartdomain_S2_spherical)

full_dynamics_visualization(tangent_bundle, initial_point, t = t, steps = steps, surface = embedding)
