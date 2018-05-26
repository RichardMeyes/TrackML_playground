import h5py
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import xgboost


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

        # get labeled data (hits and corresponding labels)
        hits = hits.values[:, :4]
        num_hits = len(hits)
        print("num detected hits:", num_hits)
        print("number of unique particles detected in this event: {0}".format(len(np.unique(truth["particle_id"]))))

        # labeling data
        print("labeling data...")
        t_start = time.time()
        training_particles = truth["particle_id"].values
        hit_labels = np.zeros(len(training_particles))
        for i in range(len(training_particles)):
            hit_labels[i] = label_lookup[str(training_particles[i])]
        t_stop = time.time()
        print("labeling done! ({0} seconds)".format(t_stop - t_start))

        # train GPC
        train_size = 1000
        test_size = 100
        data_size = train_size + test_size

        # train GPclassifier
        X_train = np.array([hits[:train_size, 1],
                            hits[:train_size, 2],
                            hits[:train_size, 3]]).T
        y_train = hit_labels[:train_size]
        X_test = np.array([hits[train_size:data_size, 1],
                           hits[train_size:data_size, 2],
                           hits[train_size:data_size, 3]]).T
        y_test = hit_labels[train_size:data_size]

        # defining classifiers
        # Gaussian Process Classifier
        # kernel = 1.0 * RBF([1.0])  # isotropic
        # kernel = 1.0 * RBF([1.0, 1.0, 1.0])  # anisotropic
        # GPC = GaussianProcessClassifier(kernel=kernel)
        # Support Vector machine
        # kernel = "rbf"
        # SVC = SVC(C=1.0, kernel=kernel)
        GBC = xgboost.XGBClassifier()

        # classification
        print("starting classifier fit...")
        t_start = time.time()
        clf = GBC
        clf.fit(X_train, y_train)
        t_stop = time.time()
        print("finished fitting after {0} seconds".format(t_stop - t_start))
        print("scoring on test data...")
        t_start = time.time()
        score = clf.score(X_test, y_test)
        t_stop = time.time()
        print("score {0} after {1} seconds".format(score, t_stop - t_start))