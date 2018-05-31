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

    # loop through events and plot trajectory of chosen particle
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        # chose specific events
        if event_id > 1000:
            break
        print("event:", event_id)

        # create figure
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.02, left=0.02, top=0.98, right=0.98)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(hits.x.values, hits.y.values, hits.z.values, lw=0, marker=',', c='k')

        # get all particles id for event
        event_particle_ids = np.unique(truth["particle_id"].values)
        # choose particle range
        id_start = 6000
        id_end = 6100
        for i_particle, particle_id in enumerate(event_particle_ids[id_start:id_end]):
            hit_ids_mask = truth["particle_id"].values == particle_id
            hit_ids = truth["hit_id"].values[hit_ids_mask]
            trajectory = np.zeros((len(hit_ids), 3))
            trajectory[:, 0] = hits.x.values[hit_ids_mask]
            trajectory[:, 1] = hits.y.values[hit_ids_mask]
            trajectory[:, 2] = hits.z.values[hit_ids_mask]

            # print("particle {0}, num_hits {1}".format(particle_id, len(hit_ids)))
            if i_particle == 0:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0, marker=',', c='r')
            else:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=1, marker=',', c='b')

        ax.set_xlabel('X (millimeters)')
        ax.set_ylabel('Y (millimeters)')
        ax.set_zlabel('Z (millimeters)')

        plt.show()
