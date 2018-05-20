import os
import numpy as np
import pandas as pd

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    # training and test data folder paths
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    num_events = 1

    # loop through events
    p_ids = np.array(0)
    p_ids = list()
    for i_event in range(num_events):
        str_i_event = str(i_event)
        if len(str_i_event) == 1:
            event_prefix = "event00000100" + str_i_event
        elif len(str_i_event) == 2:
            event_prefix = "event0000010" + str_i_event

        # read data
        hits, cells, particles, truth = load_event(os.path.join(train_data_folder_path, event_prefix))

        # print(hits.head())
        # print(cells.head())
        print(particles.head())
        # print(truth.head())

        particle_idx = np.where(particles["particle_id"].values == 4503943224754176)[0][0]
        print(particles[particle_idx:particle_idx+1])
        quit()

        particle_ids = particles["particle_id"].values
        num_particles = len(particle_ids)
        # p_ids = np.hstack((p_ids, particle_ids))
        # p_ids.append(particle_ids)

        particle_ids_t = np.unique(truth["particle_id"].values)
        num_particles_detected = len(particle_ids_t)
        p_ids.append(particle_ids_t)

        print(num_particles)
        print(num_particles_detected)
        # print(np.sum(np.isin(particle_ids, particle_ids_t)))
        print('-------------------------------')

    print(len(p_ids))
    print(len(p_ids[0]))
    print(len(p_ids[1]))

    print(np.sum(np.isin(p_ids[0], p_ids[1])))