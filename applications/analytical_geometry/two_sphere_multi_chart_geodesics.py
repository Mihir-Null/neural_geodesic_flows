"""
Ad-hoc testing of things that we've implemented so far.
"""

from json import encoder
import jax
import jax.numpy as jnp

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_S2_spherical,
    parametrization_S2_spherical,
    chartdomain_S2_stereographic,
    psi_S2_inverted_stereographic,
    phi_S2_inverted_stereographic,
    psi_S2_stereographic,
    phi_S2_stereographic,
    g_S2_stereographic,
    psi_S2_spherical,
    phi_S2_spherical,
    g_S2_spherical
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

from experimental.atlas import (
    create_coordinate_domains,
    Chart,
)

from experimental.models import (
    TangentBundle_multi_chart_atlas as TangentBundle,
)


### load the sphere data ###
size = 512
data, _ = load_dataset(name = "sphere_trajectories_train", size = size)

trajectories, times = data

"""
### create 2 coordinate domains ###

#extract and flatten positions (x, y, z)
sphere = trajectories.reshape(-1, 6)  # shape (size*time, 6)

#apply masks to get coordinate domains
domains, memberships = create_coordinate_domains(sphere,
                                                 amount_of_domains = 2,
                                                 extension_degree = 0,
                                                 is_tangent_bundle = True)

extended_upper_hemisphere, extended_lower_hemisphere = domains[0], domains[1]

#initialize chart eqx.modules
psi_extended_upper_hemisphere = psi_S2_spherical#psi_S2_inverted_stereographic
phi_extended_upper_hemisphere = phi_S2_spherical#phi_S2_inverted_stereographic
g_extended_upper_hemisphere = g_S2_spherical#g_S2_stereographic#identity_metric({'dim_M':2})

psi_extended_lower_hemisphere = psi_S2_stereographic
phi_extended_lower_hemisphere = phi_S2_stereographic
g_extended_lower_hemisphere = g_S2_stereographic#identity_metric({'dim_M':2})



### assign chart functions to the 2 coordinate domains (these need to work on those domains!) ###
chart_upper_hemisphere = Chart(coordinate_domain = extended_upper_hemisphere,
                               psi = psi_extended_upper_hemisphere,
                               phi = phi_extended_upper_hemisphere,
                               g = g_extended_upper_hemisphere)

chart_lower_hemisphere = Chart(coordinate_domain = extended_lower_hemisphere,
                               psi = psi_extended_lower_hemisphere,
                               phi = phi_extended_lower_hemisphere,
                               g = g_extended_lower_hemisphere)

sphere_atlas = (chart_upper_hemisphere, chart_lower_hemisphere)

### build a spherebundle and test global dynamics ###
sphere_bundle = TangentBundle(atlas = sphere_atlas)


"""


#define a saved model (has to be one saved in data/models/)
model_name = "multi-chart_sphere-model"

psi_initializer = NN_Jacobian_split_diffeomorphism
phi_initializer = NN_Jacobian_split_diffeomorphism
g_initializer = NN_metric

sphere_bundle = load_model(model_name,
                           psi_initializer = psi_initializer,
                           phi_initializer = phi_initializer,
                           g_initializer = g_initializer)


#initial point in the chart (consists of initial theta, phi, v^theta, v^phi)
chart_id = 0

chart = sphere_bundle.atlas[chart_id]

initial_data_point = trajectories[0,0,:]

initial_point = chart.psi(initial_data_point)


initial_state = (chart_id, initial_point)


#integration time
t = 8

#integration steps
steps = 250

#visualize the geodesic in data space and in the charts
sphere = parametrized_surface(parametrization_S2_spherical, chartdomain_S2_spherical)

full_dynamics_visualization(sphere_bundle, initial_state, t = t, steps = steps, surface = sphere)
