import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.data_utils import load_data
from utils.plot_utils import plot_dataframe
from utils.preprocessing import _align_with_earth_vertical


ROOT_DIR = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
f_s = 200.
THR_acc = 0.05
THR_gyr = 2.5

def main():
    
    # Get dataframe
    df_imu = load_data(dir_name=ROOT_DIR)
    
    # Get sensor data for left foot (LF) and right foot (RF)    
    acc_LF = df_imu[["left_foot_ACC_x", "left_foot_ACC_y", "left_foot_ACC_z"]].to_numpy()
    gyr_LF = df_imu[["left_foot_ANGVEL_x", "left_foot_ANGVEL_y", "left_foot_ANGVEL_z"]].to_numpy()
    acc_RF = df_imu[["right_foot_ACC_x", "right_foot_ACC_y", "right_foot_ACC_z"]].to_numpy()
    gyr_RF = df_imu[["right_foot_ANGVEL_x", "right_foot_ANGVEL_y", "right_foot_ANGVEL_z"]].to_numpy()
    
    # Plot 
    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(np.arange(acc_LF.shape[0])/f_s, acc_LF)
    axs[0][0].grid(which="both", axis="both", c=(0, 0, 0), alpha=0.05, ls=":")
    axs[0][0].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axs[0][0].axhline(1.0, c=(0, 0, 0), alpha=0.5, ls="--")
    
    axs[1][0].plot(np.arange(gyr_LF.shape[0])/f_s, gyr_LF[:,1], c="tab:blue", alpha=0.4, lw=4)
    axs[1][0].yaxis.set_minor_locator(plt.MultipleLocator(20.))
    axs[1][0].grid(which="both", axis="both", c=(0, 0, 0), alpha=0.05, ls=":")
    axs[1][0].sharex(axs[0][0])
    
    # Align with earth vertical
    axs[1][1].plot(np.arange(gyr_LF.shape[0])/f_s, gyr_LF[:,1], c="tab:blue", alpha=0.4, lw=4)
    acc_LF, gyr_LF = _align_with_earth_vertical(acc_LF, gyr_LF, f_s)
    
    axs[0][1].plot(np.arange(acc_LF.shape[0])/f_s, acc_LF)
    axs[0][1].grid(which="both", axis="both", c=(0, 0, 0), alpha=0.05, ls=":")
    axs[0][1].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axs[0][1].axhline(1.0, c=(0, 0, 0), alpha=0.5, ls="--")
    axs[0][1].sharex(axs[0][0])
    axs[0][1].sharey(axs[0][0])
    
    axs[1][1].plot(np.arange(gyr_LF.shape[0])/f_s, gyr_LF[:,1], c="tab:blue", lw=1)
    axs[1][1].yaxis.set_minor_locator(plt.MultipleLocator(20.))
    axs[1][1].grid(which="both", axis="both", c=(0, 0, 0), alpha=0.05, ls=":")
    axs[1][1].sharex(axs[1][0])
    axs[1][1].sharey(axs[1][0])
    
    axs[0][0].set_ylabel("acceleration, in g")
    axs[1][0].set_ylabel("angular velocity, in degrees/s")
    axs[1][0].set_xlabel("time, in s")
    axs[1][1].set_xlabel("time, in s")

    plt.show()
    return
    

if __name__ == "__main__":
    main()