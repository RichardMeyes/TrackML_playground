import os
import h5py
import json
import numpy as np

from trackml.dataset import load_event

if __name__ == "__main__":

    # training and test data folder paths
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    num_events = 100

    # # get unique list pf particle_ids
    # particle_ids = np.array(0)
    # for i_event in range(num_events):
    #     str_i_event = str(i_event)
    #     if len(str_i_event) == 1:
    #         event_prefix = "event00000100" + str_i_event
    #     elif len(str_i_event) == 2:
    #         event_prefix = "event0000010" + str_i_event
    #
    #     # read data
    #     print("reading data from {0}".format(event_prefix))
    #     _, _, particles, _ = load_event(os.path.join(train_data_folder_path, event_prefix))
    #
    #     # read all particle_ids and store them with the number of hits per event
    #     event_particle_ids = particles["particle_id"].values
    #     particle_ids = np.hstack((particle_ids, event_particle_ids))
    #
    # particle_ids = np.sort(np.unique(particle_ids))
    #
    # with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "w") as f:
    #     f.create_dataset("particle_ids", data=particle_ids)

    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    # create labels starting from 0,1,2,...
    num_unique_particles = len(particle_ids)
    particle_ids_labels = np.zeros((num_unique_particles, 2))
    particle_ids_labels[:, 0] = particle_ids
    particle_ids_labels[:, 1] = np.arange(num_unique_particles)

    # # store labels into hdf5 file
    # with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r+") as f:
    #     f.create_dataset("particle_ids_labels", data=particle_ids_labels)
    
    # save as a label_lookup
    # label data
    label_lookup = dict()
    for i in range(len(particle_ids)):
        label_lookup[str(particle_ids[i])] = int(particle_ids_labels[i, 1])
    with open("../../data/preprocessed/train_sample/lookup_particle_ids_labels.json", "w") as f:
        json.dump(label_lookup, f)