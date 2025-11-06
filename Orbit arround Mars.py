import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime, date_time_from_epoch

#By Paul L. Mennewisch

spice.load_standard_kernels()

bodies_to_create = ["Mars","Phobos", "Deimos"]

global_frame_origin = "Mars"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
   bodies_to_create, global_frame_origin, global_frame_orientation)

body_settings.add_empty_settings("Spacecraft")

bodies = environment_setup.create_system_of_bodies(body_settings)

bodies_to_propagate = ["Spacecraft"]
central_bodies = ["Mars"]

acceleration_settings_Spacecraft = dict(
   Mars=[propagation_setup.acceleration.point_mass_gravity()],
   Phobos=[propagation_setup.acceleration.point_mass_gravity()],
   Deimos=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings = {"Spacecraft": acceleration_settings_Spacecraft}

acceleration_models = propagation_setup.create_acceleration_models(
   bodies, acceleration_settings, bodies_to_propagate, central_bodies)

mars_gravitational_parameter = bodies.get("Mars").gravitational_parameter

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter = mars_gravitational_parameter,
   semi_major_axis = 8.000000e+06, # meters
   eccentricity = 0.5, # unitless
   inclination = 1e+00, # radians
   argument_of_periapsis = 2.82793255239e+00, # radians
   longitude_of_ascending_node = 2.965174772791e+00, # radians
   true_anomaly = 1.5708, # radians
)

simulation_start_epoch = DateTime(2025,11,1).epoch()#DateTime(2025, 11, 5).to_epoch()
simulation_end_epoch   = DateTime(2025,11,5).epoch()##DateTime(2025, 11, 6).to_epoch()

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
   time_step=10.0, coefficient_set=propagation_setup.integrator.rk_4
)

propagator_type = propagation_setup.propagator.cowell #(encke)

termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

dependent_variables_to_save = [
   propagation_setup.dependent_variable.latitude("Spacecraft", "Mars"),
   propagation_setup.dependent_variable.longitude("Spacecraft", "Mars")
]

propagator_settings = propagation_setup.propagator.translational(
   central_bodies,
   acceleration_models,
   bodies_to_propagate,
   initial_state,
   simulation_start_epoch,
   integrator_settings,
   termination_settings,
   propagator=propagator_type,
   output_variables=dependent_variables_to_save
)

dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
   bodies, propagator_settings
)

states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history
dependent_variables_array = result2array(dependent_variables)

print(
   f"""
Single Earth-Orbiting Satellite Example.
The initial position vector of Delfi-C3 is [km]: \n
{states[simulation_start_epoch][:3] / 1E3}
The initial velocity vector of Delfi-C3 is [km/s]: \n
{states[simulation_start_epoch][3:] / 1E3} \n
After {simulation_end_epoch - simulation_start_epoch} seconds the position vector of Delfi-C3 is [km]: \n
{states[simulation_end_epoch][:3] / 1E3}
And the velocity vector of Delfi-C3 is [km/s]: \n
{states[simulation_end_epoch][3:] / 1E3}
"""
)

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Orbit of Spacecraft arround Mars')

# Plot the positional state history
ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3], label=bodies_to_propagate[0], linestyle='-.')

planet_radius = 3396000
u = np.linspace(0, 2*np.pi, 15)
v = np.linspace(0, np.pi, 15)
u, v = np.meshgrid(u, v)
x = planet_radius * np.sin(v) * np.cos(u)
y = planet_radius * np.sin(v) * np.sin(u)
z = planet_radius * np.cos(v)
ax.plot_surface(x, y, z, color='red', alpha=0.8)

ax.set_xlim([-2*planet_radius, 2*planet_radius])
ax.set_ylim([-2*planet_radius, 2*planet_radius])
ax.set_zlim([-2*planet_radius, 2*planet_radius])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()