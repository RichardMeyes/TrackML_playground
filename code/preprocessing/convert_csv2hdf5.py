import os
import h5py
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # training and test data folder paths
    train_data_folder_path = "../../data/raw/train_sample/train_100_events"
    test_data_folder_path = "../../data/raw/test/test/"

    # loop through data folders
    for data_folder_path in [train_data_folder_path, test_data_folder_path]:
        # get a list with all data files
        f_list = os.listdir(data_folder_path)
        num_files = len(f_list)

        # create hdf5 file if it does not already exist
        fp_train_data_hdf5 = "../../data/converted/train_sample/train_100_events.hdf5"
        fp_test_data_hdf5 = "../../data/converted/test/test.hdf5"
        if data_folder_path == train_data_folder_path:
            fp_hdf5 = fp_train_data_hdf5
        elif data_folder_path == test_data_folder_path:
            fp_hdf5 = fp_test_data_hdf5

        if not os.path.isfile(fp_hdf5):
            with h5py.File(fp_hdf5, "w") as f_hdf5:
                print("created hdf5 file!")
        else:
            print("hdf5 file already exists")

        # parse .csv data to hdf5 file
        print("parsing header information to hdf5 file")
        # define number of different data files
        if data_folder_path == train_data_folder_path:
            data_chars = ["cells", "hits", "particles", "truth"]
            k = len(data_chars)
        elif data_folder_path == test_data_folder_path:
            data_chars = ["cells", "hits"]
            k = len(data_chars)

        header_info = list()
        with h5py.File(fp_hdf5, "r+") as f_hdf5:
            for i, file_name in enumerate(f_list[:k]):
                file_path = "/".join([data_folder_path, file_name])
                df = pd.read_csv(file_path, header=0)
                header_info = np.array(list(df), dtype="S")
                f_hdf5.create_dataset("_".join(["header_info", data_chars[i]]), data=header_info)
        print("parsed header information to hdf5 file")

        print("parsing .csv data to hdf5 file")
        with h5py.File(fp_hdf5, "r+") as f_hdf5:
            for i, file_name in enumerate(f_list):
                file_path = "/".join([data_folder_path, file_name])
                df = pd.read_csv(file_path, header=0)
                f_hdf5.create_dataset(file_name, data=df)
                print("parsed file {0}/{1}".format(i+1, num_files))
