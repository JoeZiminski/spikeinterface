"""
TODO: some notes on this debugging script.
"""
import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_inter_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

# Generate a ground truth recording where every unit is firing a lot,
# with high amplitude, and is close to the spike, so all picked up.
# This just makes it easy to play around with the units, e.g.
# if specifying 5 units, 5 unit peaks are clearly visible, none are lost
# because their position is too far from probe.

default_unit_params_range = dict(
    alpha=(100.0, 500.0),
    depolarization_ms=(0.09, 0.14),
    repolarization_ms=(0.5, 0.8),
    recovery_ms=(1.0, 1.5),
    positive_amplitude=(0.1, 0.25),
    smooth_ms=(0.03, 0.07),
    spatial_decay=(20, 40),
    propagation_speed=(250.0, 350.0),
    b=(0.1, 1),
    c=(0.1, 1),
    x_angle=(0, np.pi),
    y_angle=(0, np.pi),
    z_angle=(0, np.pi),
)

default_unit_params_range["alpha"] = (400, 500)  # do this or change the margin on generate_unit_locations_kwargs

rec_list, _ = generate_inter_session_displacement_recordings(
    non_rigid_gradient=None,  # 0.05,
    num_units=1,
    rec_durations=(25, 25, 25),  # TODO: checks on inputs
    rec_shifts=(
        0,
        200,
        400,
    ),
    generate_sorting_kwargs=dict(firing_rates=(149, 150), refractory_period_ms=4.0),
    generate_templates_kwargs=dict(unit_params=default_unit_params_range, ms_before=1.5, ms_after=3),
    seed=44,
    generate_unit_locations_kwargs=dict(
        margin_um=0.0,  # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
        minimum_z=5.0,
        maximum_z=45.0,
        minimum_distance=18.0,
        max_iteration=100,
        distance_strict=False,
    ),
)


for rec in rec_list:

    si.plot_traces(rec, time_range=(0, 1))
    plt.show()

    peaks = detect_peaks(rec, method="locally_exclusive")
    peak_locs = localize_peaks(rec, peaks, method="grid_convolution")

    si.plot_drift_raster_map(
        peaks=peaks,
        peak_locations=peak_locs,
        recording=rec,
        clim=(-300, 0)  # fix clim for comparability across plots
    )
    plt.show()