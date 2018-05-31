import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event


def cart2spherical(coords):
    r = np.linalg.norm(coords, axis=1)
    theta = np.degrees(np.arccos(coords[:, 2] / r))
    phi = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
    return np.vstack((r, theta, phi)).T


if __name__ == "__main__":

    # get list of all particle ids
    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    # choose single particle
    particle_id = particle_ids[2]
    print("particle_id:", particle_id)

    # create figure
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')

    # loop through events and plot trajectory of chosen particle
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        # chose specific events
        if event_id > 1000:
            break
        print("event:", event_id)

        ax.plot(hits.x.values, hits.y.values, hits.z.values, lw=0, marker=',', c='k')

    ax.set_xlabel('X (millimeters)')
    ax.set_ylabel('Y (millimeters)')
    ax.set_zlabel('Z (millimeters)')

    plt.show()
