import os
import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class Clusterer(object):

    def __init__(self, eps):
        self.eps = eps

    def _preprocess(self, hits):
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

    def predict(self, hits):
        X = self._preprocess(hits)

        cl = DBSCAN(eps=self.eps, min_samples=1, algorithm='kd_tree')
        labels = cl.fit_predict(X)

        return labels


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


if __name__ == "__main__":

    # training and test data folder paths
    path_to_train = "../data/raw/train_sample/train_100_events"

    # chose a single event to work with
    event_prefix = "event000001000"

    # read data
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

    # perform clustering
    model = Clusterer(eps=0.008)
    labels = model.predict(hits)

    print(labels)

    submission = create_one_event_submission(0, hits, labels)
    score = score_event(truth, submission)

    print("Your score: ", score)

