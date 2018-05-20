import os
import numpy as np
import pandas as pd

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

if __name__ == "__main__":

    # training and test data folder paths
    path_to_train = "../data/raw/train_sample/train_100_events"

    # chose a single event to work with
    event_prefix = "event000001010"

    # read data
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

    # print(hits.head())
    # print(cells.head())
    print(particles.head())
    # print(truth.head())


    # print(np.sum(truth["weight"]))
