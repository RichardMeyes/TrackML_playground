import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    particle_id = particle_ids[1]
    print("particle_id:", particle_id)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        print("event:", event_id)
        particle_id_idx = np.where(truth["particle_id"] == particle_id)[0]
        particle_hit_ids = particle_id_idx + 1
        if len(particle_hit_ids) == 0:
            continue
        print(particle_hit_ids)

        event_trajectory_data = truth.values[particle_id_idx][:, 2:]
        particle_idx = np.where(particles["particle_id"] == particle_id)[0][0]
        particle_origin_data = particles.values[particle_idx:particle_idx+1][0, 1:-2]

        ax.plot(event_trajectory_data[:, 0], event_trajectory_data[:, 1], event_trajectory_data[:, 2], c='b', marker='o', alpha=0.3)
        ax.scatter(particle_origin_data[0], particle_origin_data[1], particle_origin_data[2], s=32, c='r', marker='o')

        ax.set_xlabel('X (millimeters)')
        ax.set_ylabel('Y (millimeters)')
        ax.set_zlabel('Z (millimeters)')
    plt.show()
