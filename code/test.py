import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.data_utils import load_data
from utils.plot_utils import plot_dataframe
from utils.preprocessing import _transform


ROOT_DIR = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
f_s = 200.
THR_acc = 0.05
THR_gyr = 2.5

def main():
    
    # Get list of subject ids
    df_imu = load_data(dir_name=ROOT_DIR)
    
    # Plot
    # plot_dataframe(df_imu, "left_foot")
    
    accL = df_imu[["left_foot_ACC_x", "left_foot_ACC_y", "left_foot_ACC_z"]].to_numpy()
    gyrL = df_imu[["left_foot_ANGVEL_x", "left_foot_ANGVEL_y", "left_foot_ANGVEL_z"]].to_numpy()
    accR = df_imu[["right_foot_ACC_x", "right_foot_ACC_y", "right_foot_ACC_z"]].to_numpy()
    gyrR = df_imu[["right_foot_ANGVEL_x", "right_foot_ANGVEL_y", "right_foot_ANGVEL_z"]].to_numpy()
    
    # Transform to common coordinate system
    acc_L = _transform(accL, gyrL, f_s)
    acc_R = _transform(accR, gyrR, f_s)
    return
    

if __name__ == "__main__":
    main()