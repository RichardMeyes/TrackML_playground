import os
import h5py
import numpy as np
import json

from trackml.dataset import load_event

if __name__ == "__main__":

    # training and test data folder paths
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    num_events = 100

    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    num_particles = len(particle_ids)
    print("{0} unique particles found.".format(num_particles))

    # read aggregated particle data
    print("reading aggregated particle data...")
    with open("../../data/preprocessed/train_sample/aggregated_particle_data.json", "r") as f:
        particle_info = json.load(f)

    for i_event in range(num_events):
        str_i_event = str(i_event)
        if len(str_i_event) == 1:
            event_prefix = "event00000100" + str_i_event
        elif len(str_i_event) == 2:
            event_prefix = "event0000010" + str_i_event

        print("reading data from {0} ({1}/{2})".format(event_prefix, i_event+1, num_events))
        hits, cells, particles, truth = load_event(os.path.join(train_data_folder_path, event_prefix))

        # aggregate info of particles
        for i_particle, particle_id in enumerate(particle_ids):
            if i_particle % 10000 == 0:
                print("reading data from particle {0}/{1}".format(i_particle+1, num_particles))

            # check if particle is in this event
            if particle_id not in truth["particle_id"].values:
                continue
            else:
                # get hit_ids in the event
                particle_id_idx = np.where(truth["particle_id"].values == particle_id)[0]
                event_hit_ids = particle_id_idx + 1
                # enrich aggregated particle data with event hit ids
                particle_info[str(particle_id)]["hit_list"] = {}
                particle_info[str(particle_id)]["hit_list"][event_prefix] = event_hit_ids.tolist()

    # save enriched aggregated particle data
    print("saving enriched aggregated particle data...")
    with open("../../data/preprocessed/train_sample/enriched_aggregated_particle_data.json", "w") as f:
        json.dump(particle_info, f)