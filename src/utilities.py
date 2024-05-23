import os
import glob
import pandas as pd
from pathlib import Path

def get_file(f): 
    # Get the path to the directory
    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            raise Exception("********This path contains more than one file*******")


def data_load(path, derive_drive_id=None):
    # Method to load data from the path to data asset
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

    print(csv_files)
    dfs = []
    # loop over the list of csv files
    for i, f in enumerate(csv_files):
        # Check if the file name matches the expected format
        if "log_pad_as_" in f and f.endswith(".csv"):
            # read the csv file
            df = pd.read_csv(f, parse_dates=['timestamp'])

            # Create a new column and fill it with the drive number
            if derive_drive_id:
                filename = os.path.basename(f)
                drive_no = filename.split('_')[3]
                print("This is the drive number: ", drive_no)
                df['drive_no'] = int(drive_no)
            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    return df