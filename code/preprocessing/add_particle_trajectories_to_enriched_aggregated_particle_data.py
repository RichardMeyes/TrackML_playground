import h5py
import json
import numpy as np
import pandas as pd

from trackml.dataset import load_event

if __name__ == "__main__":

    # training and test data folder paths
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    num_events = 100

    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    num_particles = len(particle_ids)
    print("{0} unique particles found.".format(num_particles))

    # read enriched aggregated particle data
    print("reading enriched aggregated particle data...")
    with open("../../data/preprocessed/train_sample/enriched_aggregated_particle_data.json", "r") as f:
        particle_info = json.load(f)

    for i_event in range(num_events):
        str_i_event = str(i_event)
        if len(str_i_event) == 1:
            event_prefix = "event00000100" + str_i_event
        elif len(str_i_event) == 2:
            event_prefix = "event0000010" + str_i_event
        print("reading data from {0} ({1}/{2})".format(event_prefix, i_event + 1, num_events))

        # get true trajectories
        with h5py.File("../../data/converted/train_sample/train_100_events.hdf5", "r") as f:
            event_truth_data = f["-".join([event_prefix, "truth.csv"])][...]

        # chose particle and get particle events
        for i_particle, particle_id in enumerate(particle_ids):
            if i_particle % 10000 == 0:
                print("reading data from particle {0}/{1}".format(i_particle + 1, num_particles))

            # extract event hit ids
            if "hit_list" not in particle_info[str(particle_id)].keys():
                continue
            if event_prefix not in particle_info[str(particle_id)]["hit_list"].keys():
                continue
            particle_event_hit_ids = particle_info[str(particle_id)]["hit_list"][event_prefix]
            # get event trajectory
            particle_event_trajectory_data = event_truth_data[:, 2:][np.array(particle_event_hit_ids) - 1]
            # add trajectory data to enriched agrgegated particle info data
            if "trajectory_data" not in particle_info[str(particle_id)].keys():
                particle_info[str(particle_id)]["trajectory_data"] = {}
            particle_info[str(particle_id)]["trajectory_data"][event_prefix] = {}
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["tx"] = particle_event_trajectory_data[:,
                                                                                     0].tolist()
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["ty"] = particle_event_trajectory_data[:,
                                                                                     1].tolist()
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["tz"] = particle_event_trajectory_data[:,
                                                                                     2].tolist()
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["tpx"] = particle_event_trajectory_data[:,
                                                                                     3].tolist()
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["tpy"] = particle_event_trajectory_data[:,
                                                                                     4].tolist()
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["tpz"] = particle_event_trajectory_data[:,
                                                                                     5].tolist()
            particle_info[str(particle_id)]["trajectory_data"][event_prefix]["weights"] = particle_event_trajectory_data[:,
                                                                                      6].tolist()

    # save further enriched aggregated particle data
    print("saving further enriched aggregated particle data...")
    with open("../../data/preprocessed/train_sample/further_enriched_aggregated_particle_data.json", "w") as f:
        json.dump(particle_info, f)
