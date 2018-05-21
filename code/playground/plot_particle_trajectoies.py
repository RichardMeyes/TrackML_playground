import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    # get list of all particle ids
    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    # choose single particle
    particle_id = particle_ids[1]
    print("particle_id:", particle_id)

    # create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # loop through events and plot trajectory of chosen particle
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        # chose specific events
        if event_id > 1005:
            break
        print("event:", event_id)
        particle_id_idx = np.where(truth["particle_id"] == particle_id)[0]
        particle_hit_ids = particle_id_idx + 1  # hit_ids count starts from 1 instead of 0
        # some particles may not have been detected within the event, so no hit_ids
        if len(particle_hit_ids) == 0:
            continue
        print(particle_hit_ids)

        # get true and detected particle trajectories
        event_true_trajectory_data = truth.values[particle_id_idx][:, 2:]
        event_detected_trajectory_data = hits.values[particle_id_idx][:, 1:]
        particle_idx = np.where(particles["particle_id"] == particle_id)[0][0]
        particle_origin_data = particles.values[particle_idx:particle_idx+1][0, 1:-2]

        # plot true trajectories
        ax.plot(event_true_trajectory_data[:, 0], event_true_trajectory_data[:, 1], event_true_trajectory_data[:, 2],
                c='b', marker='o', lw=2, markersize= 8, alpha=1.0)
        ax.scatter(particle_origin_data[0], particle_origin_data[1], particle_origin_data[2], s=32, c='r', marker='o')
        # plot detected trajectories
        ax.plot(event_detected_trajectory_data[:, 0], event_detected_trajectory_data[:, 1],
                event_detected_trajectory_data[:, 2], c='orange', marker='o', lw=1, markersize=4, alpha=1.0)

        ax.set_xlabel('X (millimeters)')
        ax.set_ylabel('Y (millimeters)')
        ax.set_zlabel('Z (millimeters)')
    plt.show()
