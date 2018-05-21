import os
import json
import h5py
import numpy as np

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    # with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
    #     particle_ids = f["particle_ids"][...]
    #
    # print("reading further enriched aggregated particle data...")  # load about 6.5 GB into RAM!
    # with open("../../data/preprocessed/train_sample/further_enriched_aggregated_particle_data.json", "r") as f:
    #     particle_info = json.load(f)
    #
    # particle_id = particle_ids[1]
    # print(particle_info[str(particle_id)].keys())
    # quit()

    # loop through events
    for i_event in range(100):
        str_i_event = str(i_event)
        if len(str_i_event) == 1:
            event_prefix = "event00000100" + str_i_event
        elif len(str_i_event) == 2:
            event_prefix = "event0000010" + str_i_event

        # read data
        train_data_folder_path = "../../data/raw/train_sample/train_100_events"
        hits, cells, particles, truth = load_event(os.path.join(train_data_folder_path, event_prefix))

        print(hits.head())
        print(cells.head())
        print(particles.head())
        print(truth.head())
        quit()