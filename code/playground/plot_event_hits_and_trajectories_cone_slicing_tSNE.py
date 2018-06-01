import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


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
        fig = plt.figure(figsize=(24, 8))
        fig.subplots_adjust(bottom=0.02, left=0.02, top=0.98, right=0.98)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        fig2 = plt.figure(figsize=(24, 8))
        fig2.subplots_adjust(bottom=0.08, left=0.05, top=0.95, right=0.98)
        ax4 = fig2.add_subplot(131)
        ax5 = fig2.add_subplot(132)
        ax6 = fig2.add_subplot(133, sharex=ax5, sharey=ax5)

        # plpot hits
        spurious_hits_mask = truth["particle_id"].values == 0
        ax1.plot(hits.x.values[~spurious_hits_mask],
                 hits.y.values[~spurious_hits_mask],
                 hits.z.values[~spurious_hits_mask], lw=0, marker=',', c='k')
        ax1.plot(hits.x.values[spurious_hits_mask],
                 hits.y.values[spurious_hits_mask],
                 hits.z.values[spurious_hits_mask], lw=0, marker=',', c='r')

        # get all particles id for event
        event_particle_ids = np.unique(truth["particle_id"].values)
        # choose particle range
        id_start = 1
        id_end = 100
        for i_particle, particle_id in enumerate(event_particle_ids[id_start:id_end]):
            hit_ids_mask = truth["particle_id"].values == particle_id
            hit_ids = truth["hit_id"].values[hit_ids_mask]
            trajectory = np.zeros((len(hit_ids), 3))
            trajectory[:, 0] = hits.x.values[hit_ids_mask]
            trajectory[:, 1] = hits.y.values[hit_ids_mask]
            trajectory[:, 2] = hits.z.values[hit_ids_mask]
            # plot spurious hits and trajectories
            ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=1, marker=',', c='b')

        # plot sliced cones
        hits = hits.assign(d=np.sqrt(hits.x**2 + hits.y**2 + hits.z**2))
        hits = hits.assign(r=np.sqrt(hits.x ** 2 + hits.y**2))
        hits = hits.assign(arctan2=np.arctan2(hits.z, hits.r))

        angle = 85.0
        delta_angle = 0.1
        cone_sliced_hits = hits.loc[(hits.arctan2 > (angle - delta_angle)/180*np.pi)
                                    & (hits.arctan2 < (angle + delta_angle)/180*np.pi)]
        cone_sliced_truth = truth.loc[(hits.arctan2 > (angle - delta_angle) / 180 * np.pi)
                                      & (hits.arctan2 < (angle + delta_angle) / 180 * np.pi)]
        cone_particle_ids = np.unique(truth["particle_id"].values[np.isin(truth["hit_id"].values,
                                                                          cone_sliced_hits["hit_id"].values)])
        spurious_hits_mask = cone_sliced_truth["particle_id"].values == 0
        ax2.plot(cone_sliced_hits.x.values[~spurious_hits_mask],
                 cone_sliced_hits.y.values[~spurious_hits_mask],
                 cone_sliced_hits.z.values[~spurious_hits_mask],
                 lw=0, marker=',', c='k')
        ax2.plot(cone_sliced_hits.x.values[spurious_hits_mask],
                 cone_sliced_hits.y.values[spurious_hits_mask],
                 cone_sliced_hits.z.values[spurious_hits_mask],
                 lw=0, marker=',', c='r')

        for particle_id in cone_particle_ids:
            # skip spurious particles
            if particle_id == 0:
                continue
            hit_ids_mask = cone_sliced_truth["particle_id"].values == particle_id
            hit_ids = cone_sliced_truth["hit_id"].values[hit_ids_mask]
            trajectory = np.zeros((len(hit_ids), 3))
            trajectory[:, 0] = cone_sliced_hits.x.values[hit_ids_mask]
            trajectory[:, 1] = cone_sliced_hits.y.values[hit_ids_mask]
            trajectory[:, 2] = cone_sliced_hits.z.values[hit_ids_mask]
            ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=1, marker=',')

        # plot after transformation
        r = np.sqrt(cone_sliced_hits.x.values**2 + cone_sliced_hits.y.values**2)
        a = np.arctan2(cone_sliced_hits.y.values, cone_sliced_hits.x.values)
        s = np.sin(a)
        c = np.cos(a)
        rsc = np.column_stack([r, s, c])
        r = np.sqrt(cone_sliced_hits.x.values ** 2 + cone_sliced_hits.y.values ** 2 + cone_sliced_hits.z.values ** 2)
        theta = np.degrees(np.arccos(cone_sliced_hits.z.values / r))
        phi = np.degrees(np.arctan2(cone_sliced_hits.y.values, cone_sliced_hits.x.values))
        rtp = np.column_stack([r, theta, phi])
        ax3.plot(rsc[:, 0][~spurious_hits_mask],
                 rsc[:, 1][~spurious_hits_mask],
                 rsc[:, 2][~spurious_hits_mask], lw=0, marker='.', c='k')
        ax3.plot(rsc[:, 0][spurious_hits_mask],
                 rsc[:, 1][spurious_hits_mask],
                 rsc[:, 2][spurious_hits_mask], lw=0, marker='.', c='r')
        # ax4.plot(rtp[:, 1][~spurious_hits_mask],
        #          rtp[:, 2][~spurious_hits_mask], lw=0, marker='.', c='k', alpha=0.1)
        # ax4.plot(rtp[:, 1][spurious_hits_mask],
        #          rtp[:, 2][spurious_hits_mask], lw=0, marker='.', c='r', alpha=0.2)

        # tSNE in spherical coordinates
        use_tSNE = False
        if use_tSNE:
            tSNE = TSNE(n_components=2, perplexity=30, init='random')
            t0 = time.time()
            print("tSNE fitting...")
            Y = tSNE.fit_transform(rtp[:, 1:])
            t1 = time.time()
            print("...done! {0} seconds".format(t1 - t0))
            # ax5.plot(Y[:, 0], Y[:, 1], lw=0, c='k', marker='o')
            cl = DBSCAN(eps=0.35, min_samples=3, algorithm='kd_tree')
            t0 = time.time()
            print("DBSCAN fitting...")
            labels = cl.fit_predict(Y)
            t1 = time.time()
            print("...done! {0} seconds".format(t1 - t0))
            for label in np.unique(labels):
                label_mask = labels == label
                if label == -1:
                    ax5.plot(Y[:, 0][label_mask], Y[:, 1][label_mask], lw=0, marker='x', c='k')
                else:
                    ax5.plot(Y[:, 0][label_mask], Y[:, 1][label_mask], lw=0, marker='o')
        else:
            ss = StandardScaler()
            Y = ss.fit_transform(rtp[:, 1:])
            def custom_metric(x, y):
                d = np.sqrt((x[0] - y[0])**2 + 500*(x[1] - y[1])**2)
                return d
            cl = DBSCAN(eps=0.3, min_samples=3, algorithm='auto', metric=custom_metric)
            t0 = time.time()
            print("DBSCAN fitting...")
            labels = cl.fit_predict(Y)
            t1 = time.time()
            print("...done! {0} seconds".format(t1 - t0))
            for label in np.unique(labels):
                label_mask = labels == label
                if label == -1:
                    ax5.plot(Y[:, 0][label_mask], Y[:, 1][label_mask], lw=0, marker='x', c='k')
                else:
                    ax5.plot(Y[:, 0][label_mask], Y[:, 1][label_mask], lw=0, marker='o')

        np.random.shuffle(cone_particle_ids)
        for particle_id in cone_particle_ids:
            # skip spurious particles
            if particle_id == 0:
                continue
            hit_ids_mask = cone_sliced_truth["particle_id"].values == particle_id
            hit_ids = cone_sliced_truth["hit_id"].values[hit_ids_mask]
            trajectory = np.zeros((len(hit_ids), 3))
            trajectory[:, 0] = cone_sliced_hits.x.values[hit_ids_mask]
            trajectory[:, 1] = cone_sliced_hits.y.values[hit_ids_mask]
            trajectory[:, 2] = cone_sliced_hits.z.values[hit_ids_mask]
            # transform trajectorie hits
            trajectory_trans = np.zeros_like(trajectory)
            trajectory_trans[:, 0] = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
            trajectory_trans[:, 1] = np.sin(np.arctan2(trajectory[:, 1], trajectory[:, 0]))
            trajectory_trans[:, 2] = np.cos(np.arctan2(trajectory[:, 1], trajectory[:, 0]))
            trajectory_trans_sphe = np.zeros_like(trajectory)
            trajectory_trans_sphe[:, 0] = np.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2 + trajectory[:, 2] ** 2)
            trajectory_trans_sphe[:, 1] = np.degrees(np.arccos(trajectory[:, 2] / trajectory_trans_sphe[:, 0]))
            trajectory_trans_sphe[:, 2] = np.degrees(np.arctan2(trajectory[:, 1], trajectory[:, 0]))
            # plot
            ax3.plot(trajectory_trans[:, 0], trajectory_trans[:, 1], trajectory_trans[:, 2], lw=1, marker=',')
            ax4.plot(trajectory_trans_sphe[:, 1], trajectory_trans_sphe[:, 2], lw=0, marker='.')
            if use_tSNE:
                ax6.plot(Y[:, 0][hit_ids_mask], Y[:, 1][hit_ids_mask], lw=0, marker='o')
            else:
                ax6.plot(Y[:, 0][hit_ids_mask], Y[:, 1][hit_ids_mask], lw=0, marker='o')

        for ax in [ax1, ax2]:
            ax.set_xlabel('X (millimeters)')
            ax.set_ylabel('Y (millimeters)')
            ax.set_zlabel('Z (millimeters)')
            ax1.set_xlim(-1000, 1000)
            ax1.set_ylim(-1000, 1000)
            ax1.set_zlim(-3000, 3000)

        ax3.set_xlabel('r = x**2 + y**2 (millimeters)')
        ax3.set_ylabel('sine(arctan(y/x))')
        ax3.set_zlabel('cosine(arctan(y/x))')
        ax4.set_xlabel('theta (degrees)')
        ax4.set_ylabel('phi (degrees)')
        if use_tSNE:
            ax5.set_title(str(tSNE.kl_divergence_))
            ax6.set_title(str(tSNE.kl_divergence_))

        plt.show()
