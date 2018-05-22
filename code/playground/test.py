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
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    # train_data_folder_path = "../../data/raw/test/test"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        # chose specific events
        if event_id > 1005:
            break

        print("num_particles: {0}".format(np.unique(len(particles["particle_id"].values))))

        print(hits.head())
        print(cells.head())
        print(particles.head())
        print(truth.head())
        quit()