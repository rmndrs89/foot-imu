import os
import pandas as pd
import numpy as np
from .preprocessing import _resample

# Global sampling frequency
Fs = 200.

def load_data(dir_name):
    
    # Get a list of subject IDs
    sub_ids = [sub_id for sub_id in os.listdir(dir_name) if sub_id.startswith("sub-pp")]
    
    # Loop over the subject IDs
    for (ix_sub_id, sub_id) in enumerate(sub_ids[1:]):
        print(f"{ix_sub_id:4d}: {sub_id:s}")
    
        # Get a list of valid walking trials
        event_filenames = [fname for fname in os.listdir(os.path.join(dir_name, sub_id, "motion")) if ("_task-walk" in fname) and (fname.endswith("_events.tsv"))]
        print(event_filenames)
        # Loop over the valid walking trials
        for (ix_filename, event_filename) in enumerate(event_filenames):
            
            # Set filename of text file containing sensor data
            imu_filename = event_filename.replace("_events.tsv", "_tracksys-imu_motion.tsv")
            imu_channels_filename = imu_filename.replace("_motion.tsv", "_channels.tsv")
            
            # Load IMU data
            df_imu = pd.read_csv(os.path.join(dir_name, sub_id, "motion", imu_filename), sep="\t")
            
            # Check sampling frequency
            df_channels = pd.read_csv(os.path.join(dir_name, sub_id, "motion", imu_channels_filename), sep="\t")
            if df_channels["sampling_frequency"].iloc[0].astype("float") != Fs:
                X = df_imu.to_numpy()
                X = _resample(X, df_channels["sampling_frequency"].iloc[0].astype("float"), Fs)
                df_imu = pd.DataFrame(data=X, columns=df_imu.columns)
                del X
        break
    
    return df_imu