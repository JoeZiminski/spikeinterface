from typing import Dict
import spikeinterface as si
from pathlib import Path
from spikeinterface import BinaryRecordingExtractor
from spikeinterface import load_extractor
import sys
import importlib.util
import runpy
import numpy as np
from spikeinterface import WaveformExtractor, extract_waveforms
from spikeinterface.preprocessing import whiten


class KilosortTempWhExtractor(BinaryRecordingExtractor):
    def __init__(self, output_path: Path) -> None:
        # TODO: store opts e.g. ntb, Nbatch etc here.

        if self.has_spikeinterface(output_path):
            self.sorter_output_path = output_path / "sorter_output"

            channel_indices = self.get_channel_indices()

            original_recording = load_extractor(output_path / "spikeinterface_recording.json", base_folder=output_path)
            channel_ids = original_recording.get_channel_ids()

            # TODO: check this assumption - does KS change the scale / offset? can check
            #  by performing no processing...
            if original_recording.has_scaled():
                gain_to_uV = original_recording.get_property("gain_to_uV")[channel_indices]
                offset_to_uV = original_recording.get_property("offset_to_uV")[channel_indices]
            else:
                gain_to_uV = None
                offset_to_uV = None

            channel_locations = original_recording.get_channel_locations()

            # TODO: I think this is safe to assume as if the recording was
            # sorted then it must have a probe attached.
            probe = original_recording.get_probe()

        elif self.has_valid_sorter_output(output_path):
            self.sorter_output_path = output_path

            channel_indices = self.get_channel_indices()
            channel_ids = np.array(channel_indices, dtype=str)

            gain_to_uV = None
            offset_to_uV = None

            channel_locations = np.load(self.sorter_output_path / "channel_positions.npy")
            probe = None

        else:
            raise ValueError("")

        params = self.load_and_check_kilosort_params_file()
        temp_wh_path = Path(self.sorter_output_path) / "temp_wh.dat"

        new_channel_ids = channel_ids[channel_indices]
        new_channel_locations = channel_locations[channel_indices]

        # TODO: need to adjust probe?
        # TODO: check whether this will erroneously re-order
        # is_filtered = original_recording.is_filtered or ## params was filtering run
        super(KilosortTempWhExtractor, self).__init__(
            temp_wh_path,
            params["sample_rate"],
            params["dtype"],
            num_channels=channel_indices.size,
            t_starts=None,
            channel_ids=new_channel_ids,
            time_axis=0,
            file_offset=0,
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
            is_filtered=None,
            num_chan=None,
        )
        self.set_channel_locations(new_channel_locations)

    #   if probe:
    #      self.set_probe(probe)

    def get_channel_indices(self):
        """"""
        channel_map = np.load(self.sorter_output_path / "channel_map.npy")

        if channel_map.ndim == 2:
            channel_indices = channel_map.ravel()
        else:
            assert channel_map.ndim == 1
            channel_indices = channel_map

        return channel_indices

    def has_spikeinterface(self, path_: Path) -> bool:
        """ """
        sorter_output = path_ / "sorter_output"

        if not (path_ / "spikeinterface_recording.json").is_file() or not sorter_output.is_dir():
            return False

        return self.has_valid_sorter_output(sorter_output)

    def has_valid_sorter_output(self, path_: Path) -> bool:
        """ """
        required_files = ["temp_wh.dat", "channel_map.npy", "channel_positions.npy"]

        for filename in required_files:
            if not (path_ / filename).is_file():
                print(f"The file {filename} cannot be out in {path_}")
                return False
        return True

    def load_and_check_kilosort_params_file(self) -> Dict:
        """ """
        params = runpy.run_path(self.sorter_output_path / "params.py")

        if params["dtype"] != "int16":
            raise ValueError("The dtype in kilosort's params.py is expected" "to be `int16`.")

        return params


#        original_probe = original_recording.get_probe()
# self.set_probe(original_probe) TODO: do we need to adjust the probe? what about contact positions?

# 1) figure out metadata and casting for WaveForm Extractor
# 2) check lazyness etc.

# zero padding can just be kept. Check it plays nice with WaveformExtractor...

# TODO: add provenance
# TODO: what to do about all those zeros?

#    def get_num_samples(self):
#       """ ignore Kilosort's zero-padding """
#      return self.original_recording.get_num_samples()

# TODO: check, there must be a probe if sorting was run?
# change the wiring of the probe
# TODO: check this carefully, might be completely wrong
#      if contact_vector is not None:

# if channel_map.ndim == 2:  # kilosort > 2
#    channel_indices = channel_map.ravel()  # TODO: check multiple shanks

# self.set_channel_locations(new_channel_locations)  # TOOD: check against slice_channels

#             is_filtered=None,  # TODO: need to get from KS provenence?

# In general, do we store the full channel map in channel contacts or do we
# only save the new subset? My guess is subset for contact_positions, but full probe
# for probe. Check against slice_channels.
#             self.set_probe(probe)  # TODO: what does this mean for missing channels?

#         if channel_map.ndim == 2:  # kilosort > 2
# does kilosort > 2 store shanks differently?             channel_indices = channel_map.ravel()
from spikeinterface import extractors
from spikeinterface import postprocessing

path_ = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\code\sorter_output")  # sorter_output
recording = KilosortTempWhExtractor(path_ / "sorter_output")  # TODO: without this extra is failing

original_recording = load_extractor(path_ / "spikeinterface_recording.json", base_folder=path_)

original_recording = whiten(original_recording, dtype=np.int16, mode="local", int_scale=200)

# Okay this works as a test. Steps are to:
# 1) run sorting (script below)
# 2) load the input recording and temp_wh with data above
# 3) whiten the input recording.
# 4) correlate loaded temp_wh and pp recording - these should be highly correlated if temp_wh was loaded correctly
# 5) check all properties and scaling.

# Currently the issues discussed in #1908 are a blocker. Once a decision on this is made
# this PR can be continued with.

x = recording.get_traces(start_frame=0, end_frame=10000, return_scaled=False)  # TODO: figure scaling
y = original_recording.get_traces(start_frame=0, end_frame=10000, return_scaled=False)

"""
Generate Test Data

# Load and Preprocess
data_path = Path(r"/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/code/test_run")
output_path = Path(r"/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/code/sorter_output")

recording = se.read_spikeglx(data_path)

recording = phase_shift(recording)
recording = bandpass_filter(
    recording, freq_min=300, freq_max=6000
)
recording = common_reference(
     recording, operator="median", reference="global"
)

Kilosort2_5Sorter.set_kilosort2_5_path("/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/code/Kilosort-2.5")

ss.run_sorter("kilosort2_5",
              recording,
              output_folder=output_path,
              delete_tmp_files=False,
              delete_recording_dat=False,
              skip_kilosort_preprocessing=False,
              do_correction=False,
              scaleproc=200,
              n_jobs=1
              )
"""
