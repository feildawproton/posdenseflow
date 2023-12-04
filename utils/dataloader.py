import pandas as pd
from typing import Tuple
import copy

class DataLoader(object):
    def __make_tracks(self) -> Tuple[list, list, list]:
        unique_mmsi = self.df.MMSI.unique()
        tracks_df   = self.df.groupby(["MMSI"])
        
        tracks_mmsi = []
        tracks_dtg  = []
        tracks_flts = []
        for track in unique_mmsi:
            track_df  = tracks_df.get_group(track)
            track_dtgs = track_df[["BaseDateTime"]].to_numpy()
            track_flts = track_df[["LON", "LAT", "SOG", "COG"]].to_numpy()
            
            tracks_mmsi.append(track)
            tracks_dtg.append(track_dtgs)
            tracks_flts.append(track_flts)
        
        return (tracks_mmsi, tracks_dtg, tracks_flts)
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        
        self.tracks_mmsi, self.tracks__dtg, self.tracks_flts = self.__make_tracks()
        
    def get_tracks(self) -> Tuple[list, list, list]:
        return (copy.deepcopy(self.tracks_mmsi), copy.deepcopy(self.tracks__dtg), copy.deepcopy(self.tracks_flts))
    
if __name__ == "__main__":
    val_path = "reduced_data/train.csv"
    val_dataloader = DataLoader(val_path)
    tracks_mmsi, tracks_dtg, tracks_flts = val_dataloader.get_tracks()
    print(tracks_mmsi)
    print(tracks_dtg)
    print(tracks_flts)
    
    for ndx, track in enumerate(tracks_mmsi):
        print("track", track)
        print("dtgs", len(tracks_dtg[ndx]))
        print("flts", len(tracks_flts[ndx]))
        assert len(tracks_dtg) == len(tracks_flts)