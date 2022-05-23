import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from .plot_utils import plot_compare_before_aft

def _resample(data, fs_old, fs_new):
    try:
        (N, D) = data.shape
    except:
        N = len(data)
    
    # Original time points
    t = np.arange(N)/fs_old
    
    # Correct time points and data for NaNs
    ti = t[np.logical_not(np.any(np.isnan(data), axis=1))]
    Xi = data[np.logical_not(np.any(np.isnan(data), axis=1)),:]
    
    # Fit a linear curve to the data
    f = interp1d(ti, Xi, kind="linear", axis=0, fill_value="extrapolate")
    
    # Determine time point at which to interpolate
    tq = np.arange(N/fs_old*fs_new)/fs_new
    return f(tq)

def _transform(acc_data, gyr_data, f_s, thr_acc=0.05, thr_gyr=2.5):
    """Transforms the accelerometer and gyroscope data to a common coordinate system,
    aligned with the earth vertical and the main walking direction.

    Parameters
    ----------
    acc_data : (N, 3) numpy array
        Accelerometer data, in g, with N time steps across 3 channels.
    gyr_data : (N, 3) numpy array
        Gyroscope data, in degrees/sec, with N time steps across 3 channels.
    f_s : int, float
        Sampling frequency, in Hz.
    """
    
    # Compute signal norm for accelerometer and gyroscope signals
    accN = np.linalg.norm(acc_data, axis=1)
    gyrN = np.linalg.norm(gyr_data, axis=1)
    
    # Determine if first 0.5 sec was stationary
    if ( np.all(np.abs(accN[:int(f_s//2)] - 1) < thr_acc) ) and ( np.all(np.abs(gyrN[:int(f_s//2)]) < thr_gyr) ):
        
        # Estimate earth vertical
        z_0 = np.mean(acc_data[:int(f_s//2),:], axis=0)
        z_0 /= np.linalg.norm(z_0)
    
    # Align data with walking direction
    pca = PCA(n_components=3)
    pca.fit(np.vstack((acc_data, -acc_data)))
    n_ = pca.components_[:,-1]
    
    # Determine AP and ML unit vectors
    x_0 = np.cross(n_, z_0)
    x_0 /= np.linalg.norm(x_0)
    y_0 = np.cross(z_0, x_0)
    y_0 /= np.linalg.norm(y_0)
    
    # Establish rotation matrix
    R_0 = np.hstack((x_0.reshape(-1,1), y_0.reshape(-1,1), z_0.reshape(-1,1)))
    
    # Rotate accelerometer and gyroscope data
    acc_rot = (R_0.T @ acc_data.T).T
    gyr_rot = (R_0.T @ gyr_data.T).T
    
    # Plot signals
    plot_compare_before_aft(acc_data, acc_rot, gyr_data, gyr_rot, f_s)
    
    return acc_rot, gyr_rot
    
    