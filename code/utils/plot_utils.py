from shutil import which
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_dataframe(df, tracked_point):
    
    fig, axs = plt.subplots(2, 2, sharex=True)
    
    axs[0][0].fill_between(np.arange(0, 100), np.ones((100,))*-4, np.ones((100,))*4, fc=(0, 0, 0), alpha=0.05)
    axs[0][0].plot(df[tracked_point+"_ACC_x"], lw=1)
    axs[0][0].plot(df[tracked_point+"_ACC_y"], lw=1)
    axs[0][0].plot(df[tracked_point+"_ACC_z"], lw=1)
    axs[0][0].grid(which="both", axis="both", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[0][0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    
    axs[1][0].fill_between(np.arange(0, 100), np.ones((100,))*-300, np.ones((100,))*300, fc=(0, 0, 0), alpha=0.05)
    axs[1][0].plot(df[tracked_point+"_ANGVEL_x"], lw=1)
    axs[1][0].plot(df[tracked_point+"_ANGVEL_y"], lw=1)
    axs[1][0].plot(df[tracked_point+"_ANGVEL_z"], lw=1)
    axs[1][0].grid(which="both", axis="both", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[1][0].yaxis.set_minor_locator(plt.MultipleLocator(10.))
    
    plt.show()
    return

def plot_compare_before_aft(acc0, acc, gyr0, gyr, f_s):
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1)
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax3, sharey=ax3)    
    
    ax1.plot(np.arange(acc0.shape[0])/f_s, acc0)
    ax2.plot(np.arange(acc.shape[0])/f_s, acc[:,0], c=(1, 0, 0), lw=1)
    ax2.plot(np.arange(acc.shape[0])/f_s, acc[:,1], c=(0, 0.5, 0), lw=1)
    ax2.plot(np.arange(acc.shape[0])/f_s, acc[:,2], c=(0, 0, 1), lw=1)
    ax3.plot(np.arange(gyr0.shape[0])/f_s, gyr0)
    ax4.plot(np.arange(gyr.shape[0])/f_s, gyr[:,0], c=(1, 0, 0), alpha=0.3, lw=1)
    ax4.plot(np.arange(gyr.shape[0])/f_s, gyr[:,1], c=(0, 0.5, 0), lw=1)
    ax4.plot(np.arange(gyr.shape[0])/f_s, gyr[:,2], c=(0, 0, 1), alpha=0.3, lw=1)
    
    ax1.set_ylabel('acceleration, in g')
    ax1.grid(which="both", axis="both", c=(0, 0, 0), alpha=0.1, ls=":")
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax2.grid(which="both", axis="both", c=(0, 0, 0), alpha=0.1, ls=":")
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax3.set_ylabel('angular velocity, in deg/s')
    ax3.grid(which="both", axis="both", c=(0, 0, 0), alpha=0.1, ls=":")
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(10.))
    ax4.grid(which="both", axis="both", c=(0, 0, 0), alpha=0.1, ls=":")
    ax4.yaxis.set_minor_locator(plt.MultipleLocator(10.))
    ax3.set_xlabel('time, in s')
    ax4.set_xlabel('time, in s')
    plt.show()
    return