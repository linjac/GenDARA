# replace the string "SpatialReceiver" with "MeasurmentReceiver" in all filenames in the directory
import os

def rename_files_in_dir(directory):
    print(os.listdir(directory))
    for filename in os.listdir(directory):
        os.rename(os.path.join(directory, filename), os.path.join(directory, filename.replace("_Receiver_", "_MeasurementReceiver_")))
        
rename_files_in_dir("./")

# replace all substrings "SpatialReceiver" with "MeasurmentReceiver" in all cells in the csv file meta.csv in this directory
import pandas as pd

def rename_cells_df(directory):
    df = pd.read_csv(os.path.join(directory, "meta.csv"))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            # check if the cell is a string
            if isinstance(df.iloc[i, j], str):
                df.iloc[i, j] = df.iloc[i, j].replace("Receiver_", "MeasurementReceiver_")
    df.to_csv(os.path.join(directory, "meta.csv"), index=False)
    
# rename_cells_df("./")

