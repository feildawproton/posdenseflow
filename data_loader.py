import os
import glob
import pandas as pd
import numpy as np

class AIS_DataLoader():
    def __init__(self, data_dict: dict):
        # -- GET LIST -- #
        data_dir_abs = os.path.abspath(data_dict["data_dir"])
        search_paths = os.path.join(data_dir_abs, "AIS_*.csv")
        files = glob.glob(search_paths)
        print("Found " + str(len(files)) + " csv files.")

        # -- OPEN AND FILTER DATAFRAMES -- #
        df_list = []
        for file in files:
            print("Loading: " + file)
            this_df = pd.read_csv(file)
            if data_dict["filter_loc"] == True:
                this_df = this_df[(this_df.LON >= data_dict["min_lon"]) & (this_df.LON <= data_dict["max_lon"]) &
                                  (this_df.LAT >= data_dict["min_lat"]) & (this_df.LAT <= data_dict["max_lat"])]
            df_list.append(this_df)
            print("Done.")

        # -- CONCAT IF NEEDED -- #
        if len(df_list) > 1:
            self.df = pd.concat(df_list)
        else:
            self.df = df_list[0]    

        # -- MAKE USEFUL DTG -- #
        if "DTG" not in self.df.columns:
            print("converting times to something useful")
            self.df["DTG"] = np.array(self.df["BaseDateTime"], dtype=np.datetime64)
            self.df = self.df.drop("BaseDateTime", axis=1)

        print(self.df.head())
        print(self.df.columns)

        self.df.sort_values(by="DTG", inplace=True)

        print(self.df.head())
        print(self.df.columns)
        print(self.df["DTG"])

        # -- SAVE FILTERED IF DESIRED -- #
        if data_dict["save"] == True:
            self.df.to_csv(os.path.abspath(data_dict["save_dir"] + "AIS_filtered.csv")) 

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        return self.df.loc[[idx]]

def preprocess():
    data_dict = {}
    data_dict["data_dir"]   = "data/"
    data_dict["filter_loc"] = True
    data_dict["min_lon"]    = -77.133719
    data_dict["max_lon"]    = -74.957005
    data_dict["min_lat"]    = 36.176017
    data_dict["max_lat"]    = 37.599435
    data_dict["save"]       = True
    data_dict["save_dir"]   = "VaBeachData/"
    dataloader = AIS_DataLoader(data_dict=data_dict)

def test_dataloader():
    data_dict = {}
    data_dict["data_dir"]   = "VaBeachData/"
    data_dict["filter_loc"] = False
    data_dict["save"]       = False
    dataloader = AIS_DataLoader(data_dict=data_dict)

    print("lest test retrieving row from the dataframe")
    for i in range(len(dataloader)):
        row_i = dataloader[i]
        print(row_i)

if __name__ == "__main__":
    test_dataloader()
