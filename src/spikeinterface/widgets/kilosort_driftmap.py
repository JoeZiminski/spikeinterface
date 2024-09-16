from __future__ import annotations

import numpy as np


from .base import BaseWidget, to_attr


class KilosortDriftMap(BaseWidget):

    def __init__(
        self,
        output_folder,
        localised_spikes_only=False,
        exclude_node=True,
        **backend_kwargs,
    ):
        data_plot = dict(output_folder=output_folder)

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):

        breakpoint()

        temps = "templates.npy"
        spikeTemplates = "spike_templates.npy"
        tempScalingAmps = readNPY(fullfile(ksDir, "amplitudes.npy"))

        ss = readNPY(fullfile(ksDir, "spike_times.npy"))
        # TODO: can load into a sorting object probs easier...
        st = double(ss) / spikeStruct.sample_rate

        winv = readNPY(fullfile(ksDir, "whitening_mat_inv.npy"))

        """
        if exist(fullfile(ksDir, 'spike_clusters.npy')) OR IS EMPTY CLUSTER GROUPS FILE
            clu = readNPY(fullfile(ksDir, 'spike_clusters.npy'));
        else
            clu = spikeTemplates;
        end
        """

        # coords = readNPY(fullfile(ksDir, 'channel_positions.npy'));
        # ycoords = coords(:, 2); xcoords = coords(:, 1);

        #    pcFeat = readNPY(fullfile(ksDir,'pc_features.npy')); % nSpikes x nFeatures x nLocalChannels
        #   pcFeatInd = readNPY(fullfile(ksDir,'pc_feature_ind.npy')); % nTemplates x nLocalChannels

        if exlucde_noise:
            st = st(~ismember(clu, noiseClusters))
            spikeTemplates = spikeTemplates(~ismember(clu, noiseClusters))
            tempScalingAmps = tempScalingAmps(~ismember(clu, noiseClusters))

            # if load_pcs:
            # pcFeat = pcFeat(~ismember(clu, noiseClusters), :,:);
            clu = clu(~ismember(clu, noiseClusters))
