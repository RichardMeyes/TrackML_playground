import os
import h5py
import numpy as np
import pandas as pd

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    # training and test data folder paths
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    num_events = 100

    # get unique list pf particle_ids
    particle_ids = np.array(0)
    for i_event in range(num_events):
        str_i_event = str(i_event)
        if len(str_i_event) == 1:
            event_prefix = "event00000100" + str_i_event
        elif len(str_i_event) == 2:
            event_prefix = "event0000010" + str_i_event

        # read data
        print("reading data from {0}".format(event_prefix))
        _, _, particles, _ = load_event(os.path.join(train_data_folder_path, event_prefix))

        # read all particle_ids and store them with the number of hits per event
        event_particle_ids = particles["particle_id"].values
        particle_ids = np.hstack((particle_ids, event_particle_ids))

    particle_ids = np.sort(np.unique(particle_ids))

    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "w") as f:
        f.create_dataset("particle_ids", data=particle_ids)