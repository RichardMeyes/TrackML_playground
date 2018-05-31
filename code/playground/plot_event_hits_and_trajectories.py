import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event


def cart2spherical(coords):
    r = np.linalg.norm(coords, ax1is=1)
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
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(bottom=0.02, left=0.02, top=0.98, right=0.98)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)

        # plpot hits
        ax1.plot(hits.x.values, hits.y.values, hits.z.values, lw=0, marker=',', c='k')

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

            # plot spurious hits and trajectories
            if i_particle == 0:
                ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0, marker=',', c='r')
            else:
                ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=1, marker=',', c='b')

        # plot sliced cones
        hits = hits.assign(d=np.sqrt(hits.x**2 + hits.y**2 + hits.z**2))
        hits = hits.assign(r=np.sqrt(hits.x ** 2 + hits.y**2))
        hits = hits.assign(arctan2=np.arctan2(hits.z, hits.r))

        angle = 75
        delta_angle = 0.5
        cone_sliced_hits = hits.loc[(hits.arctan2 > (angle - delta_angle)/180*np.pi)
                                    & (hits.arctan2 < (angle + delta_angle)/180*np.pi)]

        ax2.plot(cone_sliced_hits.x.values, cone_sliced_hits.y.values, cone_sliced_hits.z.values,
                 lw=0, marker=',', c='k')

        for ax in [ax1, ax2]:
            ax.set_xlabel('X (millimeters)')
            ax.set_ylabel('Y (millimeters)')
            ax.set_zlabel('Z (millimeters)')
            ax1.set_xlim(-1000, 1000)
            ax1.set_ylim(-1000, 1000)
            ax1.set_zlim(-3000, 3000)

        plt.show()
