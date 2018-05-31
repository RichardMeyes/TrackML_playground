import h5py
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import GradientBoostingClassifier

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import xgboost


def preprocess(hits):
    x = hits.x.values
    y = hits.y.values
    z = hits.z.values

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hits['x2'] = x / r
    hits['y2'] = y / r

    r = np.sqrt(x ** 2 + y ** 2)
    hits['z2'] = z / r

    ss = StandardScaler()
    X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)

    return X


if __name__ == "__main__":

    # get list of all particle ids
    with h5py.File("../../data/preprocessed/train_sample/train_100_events.hdf5", "r") as f:
        particle_ids = f["particle_ids"][...]
        particle_ids_labels = f["particle_ids_labels"][...]

    with open("../../data/preprocessed/train_sample/lookup_particle_ids_labels.json", "r") as f:
        label_lookup = json.load(f)

    # loop through events and plot trajectory of chosen particle
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    for event_id, hits, cells, particles, truth in load_dataset(train_data_folder_path):
        # chose specific events
        if event_id > 1000:
            break

        X = preprocess(hits)
        mask = truth["particle_id"].values != 0
        X = X[mask]

        num_dp = len(X)
        print("number of data points ", num_dp)
        train_size = int(0.8*num_dp)
        test_size = num_dp - train_size

        X_train = X[:train_size]
        X_test = X[train_size:]

        print("starting classifier fit...")
        t_start = time.time()
        clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01)
        clf.fit(X_train)
        t_stop = time.time()
        print("finished fitting after {0} seconds".format(t_stop - t_start))
        print("scoring on test data...")
        t_start = time.time()
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        t_stop = time.time()
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        print("training error ", n_error_train, "test error", n_error_test)