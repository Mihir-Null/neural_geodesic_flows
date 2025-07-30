"""
This module uses the TangentBundle_multi_chart_atlas class to calculate geodesics
on the entire two sphere. It uses a two chart atlas; these are
a stereographic projection from the north and south. 
"""

from core.template_psi_phi_g_functions_analytical import (
    chartdomain_S2_spherical,
    parametrization_S2_spherical,
    psi_S2_inverted_stereographic,
    phi_S2_inverted_stereographic,
    psi_S2_stereographic,
    phi_S2_stereographic,
    g_S2_stereographic,
)

from experimental.utils import (
    load_dataset,
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


### load some sphere data to pass to the model ###
size = 64
data, _ = load_dataset(name = "sphere_trajectories_train", size = size)

trajectories, times = data

#extract and flatten positions (x, y, z)
points = trajectories.reshape(-1, 6)  # shape (size*time, 6)

#later we can chose on of those to evolve along a geodesic
initial_points = trajectories[:,0,:]

#build the two coordinate domains: extended upper and lower hemispheres
domains, memberships = create_coordinate_domains(points,
                                                 amount_of_domains = 2,
                                                 extension_degree = 0,
                                                 is_tangent_bundle = True)

extended_upper_hemisphere, extended_lower_hemisphere = domains[0], domains[1]

#initialize Chart instances for the two domains
chart_upper_hemisphere = Chart(coordinate_domain = extended_upper_hemisphere,
                               psi = psi_S2_inverted_stereographic,
                               phi = phi_S2_inverted_stereographic,
                               g = g_S2_stereographic)

chart_lower_hemisphere = Chart(coordinate_domain = extended_lower_hemisphere,
                               psi = psi_S2_stereographic,
                               phi = phi_S2_stereographic,
                               g = g_S2_stereographic)

#build the atlas
sphere_atlas = (chart_upper_hemisphere, chart_lower_hemisphere)

#initialize the model
sphere_bundle = TangentBundle(atlas = sphere_atlas)


#chose an initial point
initial_point = initial_points[16]

#integration time
t = 8

#integration steps
steps = 100

#for visualization purposes build the surface of the sphere in 3d
sphere = parametrized_surface(parametrization_S2_spherical, chartdomain_S2_spherical)

#visualize the geodesic in data space and in the charts
full_dynamics_visualization(sphere_bundle, initial_point, t = t, steps = steps, surface = sphere)
