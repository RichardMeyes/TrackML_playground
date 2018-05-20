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

    # create particle info data object
    particle_info = dict()
    for particle_id in particle_ids:
        particle_info[str(particle_id)] = {"vx": [], "vy": [], "vz": [], "px": [], "py": [], "pz": [], "q": [], "nhits": []}

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
            if particle_id not in particles["particle_id"].values:
                continue
            else:
                particle_id_idx = np.where(particles["particle_id"].values == particle_id)[0][0]
                particle_info[str(particle_id)]["vx"].append(float(particles[particle_id_idx:particle_id_idx + 1]["vx"].values[0]))
                particle_info[str(particle_id)]["vy"].append(float(particles[particle_id_idx:particle_id_idx + 1]["vy"].values[0]))
                particle_info[str(particle_id)]["vz"].append(float(particles[particle_id_idx:particle_id_idx + 1]["vz"].values[0]))
                particle_info[str(particle_id)]["px"].append(float(particles[particle_id_idx:particle_id_idx + 1]["px"].values[0]))
                particle_info[str(particle_id)]["py"].append(float(particles[particle_id_idx:particle_id_idx + 1]["py"].values[0]))
                particle_info[str(particle_id)]["pz"].append(float(particles[particle_id_idx:particle_id_idx + 1]["pz"].values[0]))
                particle_info[str(particle_id)]["q"].append(float(particles[particle_id_idx:particle_id_idx + 1]["q"].values[0]))
                particle_info[str(particle_id)]["nhits"].append(float(particles[particle_id_idx:particle_id_idx + 1]["nhits"].values[0]))

    with open("../../data/preprocessed/train_sample/aggregated_particle_data.json", "w") as f:
        json.dump(particle_info, f)