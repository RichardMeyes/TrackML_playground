import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
import hdbscan
from tqdm import tqdm
from sklearn.cluster import DBSCAN


class Clusterer(object):
    def __init__(self, rz_scales=[0.65, 0.965, 1.528]):
        self.rz_scales = rz_scales

    def _eliminate_outliers(self, labels, M):
        norms = np.zeros((len(labels)), np.float32)
        indices = np.zeros((len(labels)), np.float32)
        for i, cluster in tqdm(enumerate(labels), total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters == cluster)
            index = np.reshape(index, (index.shape[0]))
            indices[i] = len(index)
            x = M[index]
            norms[i] = self._test_quadric(x)
        threshold1 = np.percentile(norms, 90) * 5
        threshold2 = 20
        threshold3 = 7
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters == cluster] = 0

    def _test_quadric(self, x):
        Z = np.zeros((x.shape[0], 10), np.float32)
        Z[:, 0] = x[:, 0] ** 2
        Z[:, 1] = 2 * x[:, 0] * x[:, 1]
        Z[:, 2] = 2 * x[:, 0] * x[:, 2]
        Z[:, 3] = 2 * x[:, 0]
        Z[:, 4] = x[:, 1] ** 2
        Z[:, 5] = 2 * x[:, 1] * x[:, 2]
        Z[:, 6] = 2 * x[:, 1]
        Z[:, 7] = x[:, 2] ** 2
        Z[:, 8] = 2 * x[:, 2]
        Z[:, 9] = 1
        v, s, t = np.linalg.svd(Z, full_matrices=False)
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index, :]
        norm = np.linalg.norm(np.dot(Z, T), ord=2) ** 2
        return norm

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
        for i, rz_scale in enumerate(self.rz_scales):
            X[:, i] = X[:, i] * rz_scale

        return X

    def _init(self, dfh):
        dfh['r'] = np.sqrt(dfh.x ** 2 + dfh.y ** 2 + dfh.z ** 2)
        dfh['rt'] = np.sqrt(dfh.x ** 2 + dfh.y ** 2)
        dfh['a0'] = np.arctan2(dfh.y, dfh.x)
        dfh['r2'] = np.sqrt(dfh.x ** 2 + dfh.y ** 2)
        dfh['z1'] = dfh['z'] / dfh['rt']
        dz = 0.00012
        stepdz = 0.000005
        for ii in tqdm(range(24), desc="looping through angles"):
            dz = dz + ii * stepdz
            dfh['a1'] = dfh['a0'] + dz * dfh['z'] * np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1'] / dfh['z1']
            dfh['x2'] = 1 / dfh['z1']
            dfh['x3'] = dfh['x1'] + dfh['x2']
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['a1', 'z1', 'x1', 'x2', 'x3']].values)
            clusters = DBSCAN(eps=0.0035 - dz, min_samples=1, metric='manhattan', n_jobs=8).fit(dfs).labels_
            if ii == 0:
                dfh['s1'] = clusters
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            else:
                dfh['s2'] = clusters
                dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                maxs1 = dfh['s1'].max()
                cond = np.where(dfh['N2'].values > dfh['N1'].values)
                s1 = dfh['s1'].values
                s1[cond] = dfh['s2'].values[cond] + maxs1
                dfh['s1'] = s1
                dfh['s1'] = dfh['s1'].astype('int64')
                self.clusters = dfh['s1'].values
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        dz = 0.00012
        stepdz = -0.000005
        for ii in tqdm(range(24), desc="looping through angles"):
            dz = dz + ii * stepdz
            dfh['a1'] = dfh['a0'] + dz * dfh['z'] * np.sign(dfh['z'].values)
            dfh['x1'] = dfh['a1'] / dfh['z1']
            dfh['x2'] = 1 / dfh['z1']
            dfh['x3'] = dfh['x1'] + dfh['x2']
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['a1', 'z1', 'x1', 'x2', 'x3']].values)
            clusters = DBSCAN(eps=0.0035 + dz, min_samples=1, metric='manhattan', n_jobs=8).fit(dfs).labels_
            dfh['s2'] = clusters
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = dfh['s1'].max()
            cond = np.where(dfh['N2'].values > dfh['N1'].values)
            s1 = dfh['s1'].values
            s1[cond] = dfh['s2'].values[cond] + maxs1
            dfh['s1'] = s1
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values

    def predict(self, hits):
        self.clusters = self._init(hits)
        X = self._preprocess(hits)
        cl = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=7,
                             metric='braycurtis', cluster_selection_method='leaf', algorithm='best', leaf_size=50)
        labels = np.unique(self.clusters)
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)
            self._eliminate_outliers(labels, X)
            max_len = np.max(self.clusters)
            self.clusters[self.clusters == 0] = cl.fit_predict(X[self.clusters == 0]) + max_len
            labels = np.unique(self.clusters)
        return self.clusters


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


if __name__ == "__main__":

    # training and test data folder paths
    path_to_train = "../../data/raw/train_sample/train_100_events"

    # chose a single event to work with
    event_prefix = "event000001000"

    # read data
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

    model = Clusterer()
    labels = model.predict(hits)

    ######################################################################
    # single event
    submission = create_one_event_submission(0, hits, labels)
    score = score_event(truth, submission)
    print("Your score: ", score)

    # ######################################################################
    # # all events in the training data
    # dataset_submissions = []
    # dataset_scores = []
    # for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
    #     # Track pattern recognition
    #     model = Clusterer()
    #     labels = model.predict(hits)
    #
    #     # Prepare submission for an event
    #     one_submission = create_one_event_submission(event_id, hits, labels)
    #     dataset_submissions.append(one_submission)
    #
    #     # Score for the event
    #     score = score_event(truth, one_submission)
    #     dataset_scores.append(score)
    #
    #     print("Score for event %d: %.8f" % (event_id, score))
    # print('Mean score: %.8f' % (np.mean(dataset_scores)))

    ######################################################################
    # for test data
    path_to_test = "../../data/raw/test/test"
    test_dataset_submissions = []

    create_submission = True  # True for submission
    if create_submission:
        for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):
            # Track pattern recognition
            model = Clusterer()
            labels = model.predict(hits)

            # Prepare submission for an event
            one_submission = create_one_event_submission(event_id, hits, labels)
            test_dataset_submissions.append(one_submission)

            print('Event ID: ', event_id)

        # Create submission file
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission.to_csv('../../data/submission/submission.csv', index=False)

