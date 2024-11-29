from __future__ import annotations
from typing import TYPE_CHECKING

from torch.onnx.symbolic_opset11 import chunk

if TYPE_CHECKING:
    from spikeinterface.core.baserecording import BaseRecording

import numpy as np
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows, Motion, get_spatial_bins
from spikeinterface.sortingcomponents.motion.motion_interpolation import correct_motion_on_peaks

from spikeinterface.preprocessing.inter_session_alignment import alignment_utils
from spikeinterface.preprocessing.motion import run_peak_detection_pipeline_node
import copy


# 1) add docstrings and type hints
# 2) add print statements to the entry function
# 3) look into the DREDGE stuff

# TODO: 1) think of a method to choose some reasonable defaults for bin size, nonrigid smoothing.
# TODO: 2) different alignment procedure


def get_estimate_histogram_kwargs() -> dict:
    """
    A dictionary controlling how the histogram for each session is
    computed. The session histograms are estimated by chunking
    the recording into time segments and computing histograms
    for each chunk, then performing some summary statistic over
    the chunked histograms.

    Returns
    -------
    A dictionary with entries:

    "bin_um" : number of spatial histogram bins. As the estimated peak
        locations are continuous (i.e. real numbers) this is not constrained
        by the number of channels.
    "method" : may be "chunked_mean", "chunked_median", "chunked_supremum",
        "chunked_poisson". Determines the summary statistic used over
        the histograms computed across a session. See `alignment_utils.py
        for details on each method.
    "chunked_bin_size_s" : The length in seconds to chunk the recording
        for estimating the chunked histograms. If set to "estimate",
        the size is estimated from firing frequencies.
    "log_scale" : if `True`, histograms are log transformed.
    "depth_smooth_um" : if `None`, no smoothing is applied. See
        `make_2d_motion_histogram`.
    """
    return {
        "bin_um": 2,
        "method": "chunked_mean",
        "chunked_bin_size_s": "estimate",
        "log_scale": False,
        "depth_smooth_um": None,
    }


def get_compute_alignment_kwargs() -> dict:
    """
    A dictionary with settings controlling how inter-session
    alignment is estimated and computed given a set of
    session activity histograms.

    All keys except for "non_rigid_window_kwargs" determine
    how alignment is estimated, based on the kilosort ("kilosort_like"
    in spikeinterface) motion correction method. See
    `iterative_template_registration` for details.

    "non_rigid_window_kwargs" : if nonrigid alignment
        is performed, this determines the nature of the
        windows along the probe depth. See `get_spatial_windows`.
    """
    return {
        "num_shifts_block": 5,
        "interpolate": False,
        "interp_factor": 10,
        "kriging_sigma": 1,
        "kriging_p": 2,
        "kriging_d": 2,
        "smoothing_sigma_bin": 0.5,
        "smoothing_sigma_window": 0.5,
        "akima_interp_nonrigid": False,
    }


def get_non_rigid_window_kwargs():
    """
    TODO: merge with motion correction
    """
    return {
        "rigid": True,
        "win_shape": "gaussian",
        "win_step_um": 50,
        "win_scale_um": 50,
        "win_margin_um": None,
        "zero_threshold": None,
    }


def get_interpolate_motion_kwargs():
    """
    Settings to pass to `InterpolateMotionRecording"
    """
    return {"border_mode": "remove_channels", "spatial_interpolation_method": "kriging", "sigma_um": 20.0, "p": 2}


###############################################################################
# Public Entry Level Functions
###############################################################################

# TODO: add some print statements for progress


def align_sessions(
    recordings_list: list[BaseRecording],
    peaks_list: list[np.ndarray],
    peak_locations_list: list[np.ndarray],
    alignment_order: str = "to_middle",
    non_rigid_window_kwargs: dict = get_non_rigid_window_kwargs(),
    estimate_histogram_kwargs: dict = get_estimate_histogram_kwargs(),
    compute_alignment_kwargs: dict = get_compute_alignment_kwargs(),
    interpolate_motion_kwargs: dict = get_interpolate_motion_kwargs(),
) -> tuple[list[BaseRecording], dict]:
    """
    Estimate probe displacement across recording sessions and
    return interpolated, displacement-corrected recording. Displacement
    is only estimated along the "y" dimension.

    This assumes peaks and peak locations have already been computed.
    See `compute_peaks_locations_for_session_alignment` for generating
    `peaks_list` and `peak_locations_list` from a `recordings_list`.

    If a  recording in `recordings_list` is already an `InterpolateMotionRecording`,
    the displacement will be added to the existing shifts to avoid duplicate
    interpolations. Note the returned, corrected recording is a copy
    (recordings in `recording_list` are not edited in-place).

    Parameters
    ----------
    recordings_list : list[BaseRecording]
        A list of recordings to be aligned.
    peaks_list : list[np.ndarray]
        A list of peaks detected from the recordings in `recordings_list`,
        as returned from the `detect_peaks` function. Each entry in
        `peaks_list` should be from the corresponding entry in `recordings_list`.
    peak_locations_list : list[np.ndarray]
        A list of peak locations, as computed by `localize_peaks`. Each entry
        in `peak_locations_list` should be matched to the corresponding entry
        in `peaks_list` and `recordings_list`.
    alignment_order : str
        "to_middle" will align all sessions to the mean position.
        Alternatively, "to_session_N" where "N" is a session number
        will align to the Nth session.
    rigid : bool
        If `True`, estimated displacement is rigid. If `False`, nonrigid
        estimation is performed by performing rigid alignment on overlapping
        subsets of the probes "y" dimension.
    estimate_histogram_kwargs : dict
        see `get_estimate_histogram_kwargs()`
    compute_alignment_kwargs : dict
        see `get_compute_alignment_kwargs()`
    non_rigid_window_kwargs : dict
        see `get_non_rigid_window_kwargs`
    interpolate_motion_kwargs : dict
        see `get_interpolate_motion_kwargs()`

    Returns
    -------
    `corrected_recordings_list : list[BaseRecording]
        List of displacement-corrected recordings (corresponding
        in order to `recordings_list`). If an input recordings is
        an  InterpolateMotionRecording` recording, the corrected
        output recording will be a copy of the input recording with
        the additional displacement correction added.
    `motion_objects_list : list[Motion]
        List of motion objects associated with each corrected
        recording. In the case where the `recording` was an
        `InterpolateMotionRecording`, no motion object is created
        and the entry in `motion_objects_list` will be `None`.

    TODO
    extra_outputs_dict : dict
        A dictionary of outputs, including variables generated
        during the displacement estiamtion and correction.
        Also, includes an "corrected" field including
        a list of corrected `peak_locations` and activity
        histogram generated after correction.
    """
    non_rigid_window_kwargs = copy.deepcopy(non_rigid_window_kwargs)
    estimate_histogram_kwargs = copy.deepcopy(estimate_histogram_kwargs)
    compute_alignment_kwargs = copy.deepcopy(compute_alignment_kwargs)
    interpolate_motion_kwargs = copy.deepcopy(interpolate_motion_kwargs)

    _check_align_sesssions_inpus(
        recordings_list, peaks_list, peak_locations_list, alignment_order, estimate_histogram_kwargs
    )

    print("Computing a single activity histogram from each session...")

    (session_histogram_list, temporal_bin_centers_list, spatial_bin_centers, spatial_bin_edges, histogram_info_list) = (
        _compute_session_histograms(recordings_list, peaks_list, peak_locations_list, **estimate_histogram_kwargs)
    )

    print("Aligning the activity histograms across sessions...")

    contact_depths = recordings_list[0].get_channel_locations()[:, 1]  # "y" dim.

    shifts_array, non_rigid_windows, non_rigid_window_centers = _compute_session_alignment(
        session_histogram_list,
        contact_depths,
        spatial_bin_centers,
        alignment_order,
        non_rigid_window_kwargs,
        compute_alignment_kwargs,
    )
    shifts_array *= estimate_histogram_kwargs["bin_um"]

    print("Creating corrected recordings...")

    corrected_recordings_list, motion_objects_list = _create_motion_recordings(
        recordings_list, shifts_array, temporal_bin_centers_list, non_rigid_window_centers, interpolate_motion_kwargs
    )

    print("Creating corrected peak locations and histograms...")

    corrected_peak_locations_list, corrected_session_histogram_list = _correct_session_displacement(
        corrected_recordings_list, peaks_list, peak_locations_list, spatial_bin_edges, estimate_histogram_kwargs
    )

    extra_outputs_dict = {
        "shifts_array": shifts_array,
        "session_histogram_list": session_histogram_list,
        "spatial_bin_centers": spatial_bin_centers,
        "temporal_bin_centers_list": temporal_bin_centers_list,
        "non_rigid_window_centers": non_rigid_window_centers,
        "non_rigid_windows": non_rigid_windows,
        "histogram_info_list": histogram_info_list,
        "motion_objects_list": motion_objects_list,
        "corrected": {
            "corrected_peak_locations_list": corrected_peak_locations_list,
            "corrected_session_histogram_list": corrected_session_histogram_list,
        },
    }
    return corrected_recordings_list, extra_outputs_dict


def align_sessions_after_motion_correction(
    recordings_list: list[BaseRecording], motion_info_list: list[dict], align_sessions_kwargs: dict | None
) -> tuple[list[BaseRecording], list[Motion], dict]:
    """
    Convenience function to run `align_sessions` to correct for
    inter-session displacement from the outputs of motion correction.

    The estimated displacement will be added to the existing recording.

    Parameters
    ----------
    recordings_list : list[BaseRecording]
        A list of motion-corrected (`InterpolateMotionRecording`) recordings.
    motion_info_list : list[dict]
        A list of `motion_info` objects, as output from `correct_motion`.
        Each entry should correspond to a recording in `recording_list`.
    rigid: bool

    align_sessions_kwargs : dict
        A dictionary of keyword arguments passed to `align_sessions`.

    TODO
    ----
    add a test that checks the output of motion_info created
    by correct_motion is as expected.
    """
    # Check motion kwargs are the same across all recordings
    motion_kwargs_list = [info["parameters"]["estimate_motion_kwargs"] for info in motion_info_list]
    if not all(kwargs == motion_kwargs_list[0] for kwargs in motion_kwargs_list):
        raise ValueError(
            "The motion correct settings used on the `recordings_list`" "must be identical for all recordings"
        )

    motion_window_kwargs = copy.deepcopy(motion_kwargs_list[0])
    if motion_window_kwargs["direction"] != "y":
        raise ValueError("motion correct must have been performed along the 'y' dimension.")

    if align_sessions_kwargs is None:
        align_sessions_kwargs = get_compute_alignment_kwargs()

    # If motion correction was nonrigid, we must use the same settings for
    # inter-session alignment or we will not be able to add the nonrigid
    # shifts together.
    if (
        "non_rigid_window_kwargs" in align_sessions_kwargs
        and align_sessions_kwargs["non_rigid_window_kwargs"]["rigid"] is False
    ):

        if motion_window_kwargs["rigid"] is False:
            print(
                "Nonrigid inter-session alignment must use the motion correct "
                "nonrigid settings. Overwriting any passed `non_rigid_window_kwargs` "
                "with the motion object non_rigid_window_kwargs."
            )
            motion_window_kwargs.pop("method")
            motion_window_kwargs.pop("direction")
            align_sessions_kwargs = copy.deepcopy(align_sessions_kwargs)
            align_sessions_kwargs["non_rigid_window_kwargs"] = motion_window_kwargs

    return align_sessions(
        recordings_list,
        [info["peaks"] for info in motion_info_list],
        [info["peak_locations"] for info in motion_info_list],
        **align_sessions_kwargs,
    )


def compute_peaks_locations_for_session_alignment(
    recording_list: list[BaseRecording],
    detect_kwargs: dict,
    localize_peaks_kwargs: dict,
    job_kwargs: dict | None = None,
    gather_mode: str = "memory",
):
    """
    A convenience function to compute `peaks_list` and `peak_locations_list`
    from a list of recordings, for `align_sessions`.

    Parameters
    ----------
    recording_list : list[BaseRecording]
        A list of recordings to compute `peaks` and
        `peak_locations` for.
    detect_kwargs : dict
        Arguments to be passed to `detect_peaks`.
    localize_peaks_kwargs : dict
        Arguments to be passed to `localise_peaks`.
    job_kwargs : dict | None
        `job_kwargs` for `run_node_pipeline()`.
    gather_mode : str
        The mode for `run_node_pipeline()`.
    """
    if job_kwargs is None:
        job_kwargs = {}

    peaks_list = []
    peak_locations_list = []

    for recording in recording_list:
        peaks, peak_locations, _ = run_peak_detection_pipeline_node(
            recording, gather_mode, detect_kwargs, localize_peaks_kwargs, job_kwargs
        )
        peaks_list.append(peaks)
        peak_locations_list.append(peak_locations)

    return peaks_list, peak_locations_list


###############################################################################
# Private Functions
###############################################################################


def _compute_session_histograms(
    recordings_list: list[BaseRecording],
    peaks_list: list[np.ndarray],
    peak_locations_list: list[np.ndarray],
    histogram_type,  # TODO think up better names
    bin_um: float,
    method: str,
    chunked_bin_size_s: str,
    depth_smooth_um: str,
    log_scale: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, list[dict]]:
    """
    Compute a 1d activity histogram for the session. As
    sessions may be long, the approach taken is to chunk
    the recording into time segments and compute separate
    histograms for each. Then, a summary statistic is computed
    over the histograms. This accounts for periods of noise
    in the recording or segments of irregular spiking.

    Parameters
    ----------
    see `align_sessions` for `recording_list`, `peaks_list`,
    `peak_locations_list`.

    see `get_estimate_histogram_kwargs()` for all other kwargs.

    Returns
    -------

    session_histogram_list : list[np.ndarray]
        A list of activity histograms (1 x n_bins), one per session.
        This is the histogram which summarises all chunked histograms.

    temporal_bin_centers_list : list[np.ndarray]
        A list of temporal bin centers, one per session. We have one
        histogram per session, the temporal bin has 1 entry, the
        mid-time point of the session.

    spatial_bin_centers : np.ndarray
        A list of spatial bin centers corresponding to the session
        activity histograms.

    spatial_bin_edges : np.ndarray
        The corresponding spatial bin edges

    histogram_info_list : list[dict]
        A list of extra information on the histograms generation
        (e.g. chunked histograms). One per session. See
        `_get_single_session_activity_histogram()` for details.
    """
    # Get spatial windows and estimate the session histograms
    temporal_bin_centers_list = []

    spatial_bin_centers, spatial_bin_edges, _ = get_spatial_bins(
        recordings_list[0], direction="y", hist_margin_um=0, bin_um=bin_um
    )

    session_histogram_list = []
    histogram_info_list = []

    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        session_hist, temporal_bin_centers, histogram_info = _get_single_session_activity_histogram(
            recording,
            peaks,
            peak_locations,
            histogram_type,
            spatial_bin_edges,
            method,
            log_scale,
            chunked_bin_size_s,
            depth_smooth_um,
        )
        temporal_bin_centers_list.append(temporal_bin_centers)
        session_histogram_list.append(session_hist)
        histogram_info_list.append(histogram_info)

    return (
        session_histogram_list,
        temporal_bin_centers_list,
        spatial_bin_centers,
        spatial_bin_edges,
        histogram_info_list,
    )


def _get_single_session_activity_histogram(
    recording: BaseRecording,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    histogram_type,
    spatial_bin_edges: np.ndarray,
    method: str,
    log_scale: bool,
    chunked_bin_size_s: float,
    depth_smooth_um: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute an activity histogram for a single session.
    The recording is chunked into time segments, histograms
    estimated and a summary statistic calculated across the histograms

    Note if `chunked_bin_size_is` is set to `"estimate"` the
    histogram for the entire session is first created to get a good
    estimate of the firing rates. TODO: this is probably overkill.
    The firing rates are used to use a time segment size that will
    allow a good estimation of the firing rate.

    Parameters
    ----------
    `spatial_bin_edges : np.ndarray
        The spatial bin edges for the created histogram. This is
        explicitly required as for inter-session alignment, the
        session histograms must share bin edges.

    see `_compute_session_histograms()` for all other keyword arguments.

    Returns
    -------
    session_histogram : np.ndarray
        Summary activity histogram for the session.
    temporal_bin_centers : np.ndarray
        Temporal bin center (session mid-point as we only have
        one time point) for the session.
    histogram_info : dict
        A dict of additional info including:
            "chunked_histograms" : The chunked histograms over which
                the summary histogram was calculated.
            "chunked_temporal_bin_centers" : The temporal vin centers
                for the chunked histograms, with length num_chunks.
            "session_std" : The mean across bin-wise standard deviation
                of the chunked histograms.
            "chunked_bin_size_s" : time of each chunk used to
                calculate the chunked histogram.
    """
    times = recording.get_times()
    temporal_bin_centers = np.atleast_1d((times[-1] + times[0]) / 2)

    # Estimate a entire session histogram if requested or doing
    # full estimation for chunked bin size
    if chunked_bin_size_s == "estimate":

        one_bin_histogram, _, _ = alignment_utils.get_activity_histogram(
            recording,
            peaks,
            peak_locations,
            spatial_bin_edges,
            log_scale=False,
            bin_s=None,
            depth_smooth_um=None,
            scale_to_hz=False,
        )

        # It is important that the passed histogram is scaled to firing rate in Hz
        scaled_hist = one_bin_histogram / recording.get_duration()
        chunked_bin_size_s = alignment_utils.estimate_chunk_size(scaled_hist)
        chunked_bin_size_s = np.min([chunked_bin_size_s, recording.get_duration()])

    if histogram_type == "1Dy":  # TODO: tidy this up

        chunked_histograms, chunked_temporal_bin_centers, _ = alignment_utils.get_activity_histogram(
            recording,
            peaks,
            peak_locations,
            spatial_bin_edges,
            log_scale,
            bin_s=chunked_bin_size_s,
            depth_smooth_um=depth_smooth_um,
            scale_to_hz=True,
        )
        if method == "chunked_mean":
            session_histogram, variation = alignment_utils.get_chunked_hist_mean(chunked_histograms)

        elif method == "chunked_median":
            session_histogram, variation = alignment_utils.get_chunked_hist_median(chunked_histograms)

        elif method == "chunked_supremum":
            session_histogram, variation = alignment_utils.get_chunked_hist_supremum(chunked_histograms)

        elif method == "chunked_poisson":
            session_histogram, variation = alignment_utils.get_chunked_hist_poisson_estimate(chunked_histograms)

        elif method == "first_eigenvector":
            session_histogram, variation = alignment_utils.get_chunked_hist_eigenvector(chunked_histograms)

        elif method == "chunked_gp":  # TODO: better name
            session_histogram, variation, gp_model = alignment_utils.get_chunked_gaussian_process_regression(
                chunked_histograms
            )

        session_variation = np.mean(
            variation
        )  # each b in independent, I think this is fine irrespective of method used

    elif histogram_type in ["2Dy_amplitude", "2Dy_x"]:

        if histogram_type == "2Dy_amplitude":
            from spikeinterface.sortingcomponents.motion.motion_utils import make_3d_motion_histograms

            chunked_histograms, chunked_temporal_bin_edges, _ = make_3d_motion_histograms(  # TODO: compute centers
                recording,
                peaks,
                peak_locations,
                direction="y",
                bin_s=chunked_bin_size_s,
                bin_um=None,
                hist_margin_um=50,
                num_amp_bins=20,  #
                log_transform=log_scale,
                spatial_bin_edges=spatial_bin_edges,
            )

        else:
            min_x = np.min(peak_locations["x"])
            max_x = np.max(peak_locations["x"])

            # TODO: add to "make_2d_histogram"

            num_x_bins = 20  # guess
            x_bins = np.linspace(min_x, max_x, num_x_bins)

            # basically direct copy from make_3d_motion_histograms
            n_samples = recording.get_num_samples()
            mint_s = recording.sample_index_to_time(0)
            maxt_s = recording.sample_index_to_time(n_samples - 1)
            bin_s = chunked_bin_size_s
            chunked_temporal_bin_edges = np.arange(mint_s, maxt_s + bin_s, bin_s)

            arr = np.zeros((peaks.size, 3), dtype="float64")
            arr[:, 0] = recording.sample_index_to_time(peaks["sample_index"])
            arr[:, 1] = peak_locations["y"]
            arr[:, 2] = peak_locations["x"]

            chunked_histograms, _ = np.histogramdd(arr, (chunked_temporal_bin_edges, spatial_bin_edges, x_bins))

        import matplotlib.pyplot as plt

        if False:
            fig, ax = plt.subplots()
            im = ax.imshow(chunked_histograms[0, :, :], origin="lower", cmap="Blues", aspect="auto")

            def update(frame):
                im.set_data(chunked_histograms[frame, :, :])
                ax.set_title(f"Slice {frame}")
                return [im]

            from matplotlib.animation import FuncAnimation

            ani = FuncAnimation(fig, update, frames=chunked_histograms.shape[0], interval=100)
            plt.show()

        session_histogram, variation = alignment_utils.get_chunked_hist_mean(chunked_histograms)

        session_histogram, variation = alignment_utils.get_chunked_hist_median(chunked_histograms)

        session_histogram, variation = alignment_utils.get_chunked_hist_supremum(chunked_histograms)

        # todo: assign these above
        num_hist = chunked_histograms.shape[0]
        num_spat_bin = chunked_histograms.shape[1]
        num_amp_bin = chunked_histograms.shape[2]

        # lol it's so mad this works
        new_chunked_hist = np.reshape(chunked_histograms, (num_hist, num_spat_bin * num_amp_bin))

        session_histogram, variation = alignment_utils.get_chunked_hist_eigenvector(
            new_chunked_hist
        )  # TODO: might want more stable method that my lazy way

        session_histogram = np.reshape(session_histogram, (num_spat_bin, num_amp_bin))
        variation = np.reshape(variation, (num_spat_bin, num_amp_bin))

        if False:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            x = spatial_bin_edges[:-1]
            y = np.arange(20)  # TODO: as above
            X, Y = np.meshgrid(x, y)

            means = session_histogram
            upper = session_histogram + variation
            lower = session_histogram - variation

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

            # Plot the mean surface
            ax.plot_surface(X, Y, variation.T, cmap="viridis", alpha=0.8, label="Mean")
            ax.plot_surface(X, Y, -variation.T, cmap="viridis", alpha=0.8, label="Mean")

            # Add labels
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            ax.set_title("Mean ± Variation")

            plt.show()

        if False:
            if method == "chunked_mean":
                session_histogram, variation = alignment_utils.get_chunked_hist_mean_2d(chunked_histograms)

            elif method == "chunked_median":
                session_histogram, variation = alignment_utils.get_chunked_hist_median_2d(chunked_histograms)

            elif method == "chunked_supremum":
                session_histogram, variation = alignment_utils.get_chunked_hist_supremum_2d(chunked_histograms)

            elif method == "first_eigenvector":
                session_histogram, variation = alignment_utils.get_chunked_hist_eigenvector_2d(chunked_histograms)

            elif method == "chunked_gp":  # TODO: better name
                session_histogram = variation = gp_model = None

        plt.imshow(session_histogram, origin="lower", cmap="Blues", aspect="auto")
        plt.title("Summary Histogram")
        plt.xlabel("Amplitude bin")
        plt.ylabel("Depth (um)")
        plt.show()

        plt.imshow(variation, origin="lower", cmap="Blues", aspect="auto")
        plt.title("Variation")
        plt.xlabel("Amplitude bin")
        plt.ylabel("Depth (um)")
        plt.show()

        session_variation = np.mean(variation)  # think about meaning of this 1d vs. 2D
        # TODO: sort out
        chunked_temporal_bin_centers = alignment_utils.get_bin_centers(chunked_temporal_bin_edges)

    histogram_info = {
        "chunked_histograms": chunked_histograms,
        "chunked_temporal_bin_centers": chunked_temporal_bin_centers,
        "session_variation": session_variation,
        "chunked_bin_size_s": chunked_bin_size_s,
        "session_histogram_variation": variation,
    }

    if method == "chunked_gp":
        histogram_info.update({"gp_model": gp_model})

    return session_histogram, temporal_bin_centers, histogram_info


def _create_motion_recordings(
    recordings_list: list[BaseRecording],
    shifts_array: np.ndarray,
    temporal_bin_centers_list: list[np.ndarray],
    non_rigid_window_centers: np.ndarray,
    interpolate_motion_kwargs: dict,
) -> tuple[list[BaseRecording], list[Motion]]:
    """
    Given a set of recordings, motion shifts and bin information per-recording,
    generate an InterpolateMotionRecording. If the recording is already an
    InterpolateMotionRecording, then the shifts will be added to a copy
    of it. Copies of the Recordings are made, nothing is changed in-place.

    Parameters
    ----------
    shifts_array : num_sessions x num_nonrigid bins

    Returns
    -------
    corrected_recordings_list : list[BaseRecording]
        A list of InterpolateMotionRecording recordings of shift-corrected
        recordings corresponding to `recordings_list`.

    motion_objects_list : list[Motion]
        A list of Motion objects. If the recording in `recordings_list`
        is already an InterpolateMotionRecording, this will be `None`, as
        no motion object is created (the existing motion object is added to)
    """
    assert all(array.ndim == 1 for array in shifts_array), "time dimension should be 1 for session displacement"

    corrected_recordings_list = []
    motion_objects_list = []
    for ses_idx, recording in enumerate(recordings_list):

        session_shift = shifts_array[ses_idx][np.newaxis, :]

        if isinstance(recording, InterpolateMotionRecording):

            corrected_recording = _add_displacement_to_interpolate_recording(
                recording, session_shift, non_rigid_window_centers
            )
            motion_objects_list.append(None)
        else:
            motion = Motion(
                [session_shift], [temporal_bin_centers_list[ses_idx]], non_rigid_window_centers, direction="y"
            )
            corrected_recording = InterpolateMotionRecording(recording, motion, **interpolate_motion_kwargs)
            motion_objects_list.append(motion)

        corrected_recordings_list.append(corrected_recording)

    return corrected_recordings_list, motion_objects_list


def _add_displacement_to_interpolate_recording(
    recording: BaseRecording,
    shifts_to_add: np.ndarray,
    new_non_rigid_window_centers: np.ndarray,
):
    """
    This function adds a shift to an InterpolateMotionRecording.

    There are four cases:
    - The original recording is rigid and new shift is rigid (shifts are added).
    - The original recording is rigid and new shifts are non-rigid (sets the
      non-rigid shifts onto the recording, then adds back the original shifts).
    - The original recording is nonrigid and the new shifts are rigid (rigid
      shift added to all nonlinear shifts)
    - The original recording is nonrigid and the new shifts are nonrigid
      (respective non-rigid shifts are added, must have same number of
      non-rigid windows).

    Parameters
    ----------
    see `_create_motion_recordings()`

    Returns
    -------
    corrected_recording : InterpolateMotionRecording
        A copy of the `recording` with new shifts added.

    TODO
    ----
    Check + ask Sam if any other fields need to be changed. This is a little
    hairy (4 possible combinations of new and old displacement shapes,
    rigid or nonrigid, so test thoroughly.
    """
    # Everything is done in place, so keep a short variable
    # name reference to the new recordings `motion` object
    # and update it.okay
    corrected_recording = copy.deepcopy(recording)

    motion_ref = corrected_recording._recording_segments[0].motion
    recording_bins = motion_ref.displacement[0].shape[1]

    # If the new displacement is a scalar (i.e. rigid),
    # just add it to the existing displacements
    if shifts_to_add.shape[1] == 1:
        motion_ref.displacement[0] += shifts_to_add[0, 0]

    else:
        if recording_bins == 1:
            # If the new displacement is nonrigid (multiple windows) but the motion
            # recording is rigid, we update the displacement at all time bins
            # with the new, nonrigid displacement added to the old, rigid displacement.
            num_time_bins = motion_ref.displacement[0].shape[0]
            tiled_nonrigid_displacement = np.repeat(shifts_to_add, num_time_bins, axis=0)
            shifts_to_add = tiled_nonrigid_displacement + motion_ref.displacement

            motion_ref.displacement = shifts_to_add
            motion_ref.spatial_bins_um = new_non_rigid_window_centers
        else:
            # Otherwise, if both the motion and new displacement are
            # nonrigid, we need to make sure the nonrigid windows
            # match exactly.
            assert np.array_equal(motion_ref.spatial_bins_um, new_non_rigid_window_centers)
            assert motion_ref.displacement[0].shape[1] == shifts_to_add.shape[1]

            motion_ref.displacement[0] += shifts_to_add

    return corrected_recording


def _correct_session_displacement(
    recordings_list: list[BaseRecording],
    peaks_list: list[np.ndarray],
    peak_locations_list: list[np.ndarray],
    spatial_bin_edges: np.ndarray,
    estimate_histogram_kwargs: dict,
):
    """
    Internal function to apply the correction from `align_sessions`
    to build a corrected histogram for comparison. First, create
    new shifted peak locations. Then, create a new 'corrected'
    activity histogram from the new peak locations.

    Parameters
    ----------
    see `align_sessions()` for parameters.

    Returns
    -------
    corrected_peak_locations_list : list[np.ndarray]
        A list of peak locations corrected by the inter-session
        shifts (one entry per session).
    corrected_session_histogram_list : list[np.ndarray]
        A list of histograms calculated from the corrected peaks (one per session).
    """
    # Correct the peak locations
    corrected_peak_locations_list = []

    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        corrected_peak_locs = correct_motion_on_peaks(
            peaks,
            peak_locations,
            recording._recording_segments[
                0
            ].motion,  # TODO: this is wrong, if the previous recording was a motion correction
            # then this will add the original motion correct and the new one. We need to pass just the new shifts
            recording,
        )
        corrected_peak_locations_list.append(corrected_peak_locs)

    corrected_session_histogram_list = []

    for recording, peaks, corrected_locations in zip(recordings_list, peaks_list, corrected_peak_locations_list):
        session_hist, _, _ = _get_single_session_activity_histogram(
            recording,
            peaks,
            corrected_locations,
            estimate_histogram_kwargs["histogram_type"],
            spatial_bin_edges,
            estimate_histogram_kwargs["method"],
            estimate_histogram_kwargs["log_scale"],
            estimate_histogram_kwargs["chunked_bin_size_s"],
            estimate_histogram_kwargs["depth_smooth_um"],
        )
        corrected_session_histogram_list.append(session_hist)

    return corrected_peak_locations_list, corrected_session_histogram_list


def _compute_session_alignment(
    session_histogram_list: list[np.ndarray],
    contact_depths: np.ndarray,
    spatial_bin_centers: np.ndarray,
    alignment_order: str,
    non_rigid_window_kwargs: dict,
    compute_alignment_kwargs: dict,
) -> tuple[np.ndarray, ...]:
    """
    Given a list of activity histograms (one per session) compute
    rigid or non-rigid set of shifts (one per session) that will bring
    all sessions into alignment.

    For rigid shifts, a cross-correlation between activity
    histograms is performed. For non-rigid shifts, the probe
    is split into segments, and linear estimation of shift
    performed for each segment.

    Parameters
    ----------
    See `align_sessions()` for parameters

    Returns
    -------
    shifts : np.ndarray
        A (num_sessions x num_rigid_windows) array of shifts to bring
        the histograms in `session_histogram_list` into alignment.
    non_rigid_windows : np.ndarray
        An array (num_non_rigid_windows x num_spatial_bins) of weightings
        for each bin in each window. For rect, these are in the range [0, 1],
        for Gaussian these are gaussian etc.
    non_rigid_window_centers : np.ndarray
        The centers (spatial, in um) of each non-rigid window.
    """
    session_histogram_array = np.array(session_histogram_list)

    akima_interp_nonrigid = compute_alignment_kwargs.pop("akima_interp_nonrigid")

    non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
        contact_depths, spatial_bin_centers, **non_rigid_window_kwargs
    )

    rigid_shifts = _estimate_rigid_alignment(
        session_histogram_array,
        alignment_order,
        compute_alignment_kwargs,
    )

    if non_rigid_window_kwargs["rigid"]:
        return rigid_shifts, non_rigid_windows, non_rigid_window_centers

    # For non-rigid, first shift the histograms according to the rigid shift
    shifted_histograms = np.zeros_like(session_histogram_array)
    for ses_idx, orig_histogram in enumerate(session_histogram_array):

        shifted_histogram = alignment_utils.shift_array_fill_zeros(
            array=orig_histogram, shift=int(rigid_shifts[ses_idx, 0])
        )
        shifted_histograms[ses_idx, :] = shifted_histogram

    # Then compute the nonrigid shifts
    nonrigid_session_offsets_matrix = alignment_utils.compute_histogram_crosscorrelation(
        shifted_histograms, non_rigid_windows, **compute_alignment_kwargs
    )
    non_rigid_shifts = _get_shifts_from_session_matrix(alignment_order, nonrigid_session_offsets_matrix)

    # Akima interpolate the nonrigid bins if required.
    if akima_interp_nonrigid:
        interp_nonrigid_shifts = _akima_interpolate_nonrigid_shifts(
            non_rigid_shifts, non_rigid_window_centers, spatial_bin_centers
        )
        shifts = rigid_shifts + interp_nonrigid_shifts
        non_rigid_window_centers = spatial_bin_centers
    else:
        shifts = rigid_shifts + non_rigid_shifts

    return shifts, non_rigid_windows, non_rigid_window_centers


def _estimate_rigid_alignment(
    session_histogram_array: np.ndarray,
    alignment_order: str,
    compute_alignment_kwargs: dict,
):
    """
    Estimate the rigid alignment from a set of activity
    histograms, using simple cross-correlation.

    Parameters
    ----------
    session_histogram_array : np.ndarray
        A (num_sessions x num_spatial_bins)  array of activity
        histograms to align
    alignment_order : str
        Align "to_middle" or "to_session_N" (where "N" is the session number)
    compute_alignment_kwargs : dict
        See `get_compute_alignment_kwargs()`.

    Returns
    -------
    optimal_shift_indices : np.ndarray
        An array (num_sessions x 1) of shifts to bring all
        session histograms into alignment.
    """
    compute_alignment_kwargs = copy.deepcopy(compute_alignment_kwargs)
    compute_alignment_kwargs["num_shifts_block"] = False

    rigid_window = np.ones_like(session_histogram_array[0, :])[np.newaxis, :]

    rigid_session_offsets_matrix = alignment_utils.compute_histogram_crosscorrelation(
        session_histogram_array,
        rigid_window,
        **compute_alignment_kwargs,
    )
    optimal_shift_indices = _get_shifts_from_session_matrix(alignment_order, rigid_session_offsets_matrix)
    return optimal_shift_indices


def _akima_interpolate_nonrigid_shifts(
    non_rigid_shifts: np.ndarray,
    non_rigid_window_centers: np.ndarray,
    spatial_bin_centers: np.ndarray,
):
    """
    Perform Akima spline interpolation on a set of non-rigid shifts.
    The non-rigid shifts are per segment of the probe, each segment
    containing a number of channels. Interpolating these non-rigid
    shifts to the spatial bin centers gives a more accurate shift
    per channel.

    Parameters
    ----------
    non_rigid_shifts : np.ndarray
    non_rigid_window_centers : np.ndarray
    spatial_bin_centers : np.ndarray

    Returns
    -------
    interp_nonrigid_shifts : np.ndarray
        An array (length num_spatial_bins) of shifts
        interpolated from the non-rigid shifts.

    TODO
    ----
    requires scipy 14
    """
    from scipy.interpolate import Akima1DInterpolator

    x = non_rigid_window_centers
    xs = spatial_bin_centers

    num_sessions = non_rigid_shifts.shape[0]
    num_bins = spatial_bin_centers.shape[0]

    interp_nonrigid_shifts = np.zeros((num_sessions, num_bins))
    for ses_idx in range(num_sessions):

        y = non_rigid_shifts[ses_idx]
        y_new = Akima1DInterpolator(x, y, method="akima", extrapolate=True)(xs)
        interp_nonrigid_shifts[ses_idx, :] = y_new

    return interp_nonrigid_shifts


def _get_shifts_from_session_matrix(alignment_order: str, session_offsets_matrix: np.ndarray):
    """
    Given a matrix of displacements between all sessions, find the
    shifts (one per session) to bring the sessions into alignment.

    Parameters
    ----------
    alignment_order : "to_middle" or "to_session_X" where
        "N" is the number of the session to align to.
    session_offsets_matrix : np.ndarray
        The num_sessions x num_sessions symmetric matrix
        of displacements between all sessions, generated by
        `_compute_session_alignment()`.

    Returns
    -------
    optimal_shift_indices : np.ndarray
        A 1 x num_sessions array of shifts to apply to
        each session in order to bring all sessions into
        alignment.
    """
    if alignment_order == "to_middle":
        optimal_shift_indices = -np.mean(session_offsets_matrix, axis=0)
    else:
        ses_idx = int(alignment_order.split("_")[-1]) - 1
        optimal_shift_indices = -session_offsets_matrix[ses_idx, :, :]

    return optimal_shift_indices


# -----------------------------------------------------------------------------
# Checkers
# -----------------------------------------------------------------------------


def _check_align_sesssions_inpus(
    recordings_list: list[BaseRecording],
    peaks_list: list[np.ndarray],
    peak_locations_list: list[np.ndarray],
    alignment_order: str,
    estimate_histogram_kwargs: dict,
):
    """
    Perform checks on the input of `align_sessions()`
    """
    num_sessions = len(recordings_list)

    if len(peaks_list) != num_sessions or len(peak_locations_list) != num_sessions:
        raise ValueError(
            "`recordings_list`, `peaks_list` and `peak_locations_list` "
            "must be the same length. They must contains list of corresponding "
            "recordings, peak and peak location objects."
        )

    if not all(rec.get_num_segments() == 1 for rec in recordings_list):
        raise ValueError(
            "Multi-segment recordings not supported. All recordings " "in `recordings_list` but have only 1 segment."
        )

    channel_locs = [rec.get_channel_locations() for rec in recordings_list]
    if not all(np.array_equal(locs, channel_locs[0]) for locs in channel_locs):
        raise ValueError(
            "The recordings in `recordings_list` do not all have "
            "the same channel locations. All recordings must be "
            "performed using the same probe."
        )

    accepted_hist_methods = ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"]
    method = estimate_histogram_kwargs["method"]
    if method not in [
        "entire_session",
        "chunked_mean",
        "chunked_median",
        "chunked_supremum",
        "chunked_poisson",
        "first_eigenvector",
        "chunked_gp",
    ]:
        raise ValueError(f"`method` option must be one of: {accepted_hist_methods}")

    if alignment_order != "to_middle":

        split_name = alignment_order.split("_")
        if not "_".join(split_name[:2]) == "to_session":
            raise ValueError(
                "`alignment_order` must take the form 'to_sesion_X'" "where X is the session number to align to."
            )

        ses_num = int(split_name[-1])
        if ses_num > num_sessions:
            raise ValueError(
                f"`alignment_order` session {ses_num} is larger than" f"the number of sessions in `recordings_list`."
            )

        if ses_num == 0:
            raise ValueError("`alignment_order` required the session number, " "not session index.")
