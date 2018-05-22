import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    # get list of all particle ids
    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]

    # choose single particle
    particle_id = particle_ids[1]
    print("particle_id:", particle_id)

    # loop through events and plot trajectory of chosen particle
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        # chose specific events
        if event_id > 1000:
            break

        # get labeled data (hits and corresponding labels)
        hits = hits.values[:, :4]
        num_hits = len(hits)
        print(num_hits)

        # get labels
        labeled_hits = truth.values[:, :2]
        # find minimun value of class label
        classes = np.unique(labeled_hits[:, 1])
        min = np.sort(classes)[1]
        # adjust all labels that are not 0
        adjust_idx = np.where(labeled_hits[:, 1] != 0)
        labeled_hits[adjust_idx, 1] -= min+1

        train_size = 1000
        test_size = 500
        data_size = train_size+test_size

        # train GPclassifier
        # training set are the first 100k hits
        X_train = np.array([hits[:train_size, 1], hits[:train_size, 2], hits[:train_size, 3]]).T
        y_train = labeled_hits[:train_size, 1].astype(int)
        # test set are the rest hits 20393 hits
        X_test = np.array([hits[train_size:data_size, 1], hits[train_size:data_size, 2], hits[train_size:data_size, 3]]).T
        y_test = labeled_hits[train_size:data_size, 1].astype(int)

        print("starting GPC fit...")
        t_start = time.time()
        kernel = 1.0 * RBF([1.0])  # isotropic
        kernel = 1.0 * RBF([1.0, 1.0, 1.0])  # anisotropic
        GPc = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        t_stop = time.time()
        print("finished fitting after {0} seconds".format(t_stop-t_start))
        print("scoring on test data...")
        t_start = time.time()
        score = GPc.score(X_test, y_test)
        t_stop = time.time()
        print("score {0} after {1} seconds".format(score, t_stop-t_start))