import pandas as pd
from typing import Tuple
#import copy
import numpy as np
#from numba import njit
import time
#import multiprocessing as mp

# expects the format [samples, [time, x, y]]
#@njit is slower don't do it
'''
def __calc_vels(TXY: np.ndarray) -> np.ndarray:
    if TXY.shape[0] >= 2:
        dTXY   = TXY[1:, :] - TXY[:-1, :]
        dXY_dT = np.divide(dTXY[:, 1:], np.expand_dims(dTXY[:, 0], axis=-1))
        return dXY_dT
    else:
        return np.zeros((1,2))
'''    
def __calc_vels(T: np.ndarray, XY: np.ndarray) -> np.ndarray:
    if T.shape[0] >= 2:
        dT  = np.subtract(T[1:]    , T[:-1]    ) 
        print(dT)
        dXY = np.subtract(XY[1:, :], XY[:-1, :]) 
        dXY_dT = np.divide(dXY, np.expand_dims(dT, axis=-1))
        return dXY_dT
    else:
        return np.zeros((1,2))

def __get_tracks(df: pd.DataFrame) -> Tuple[list, list, float]:
    unique_mmsi = df.MMSI.unique()
    tracks_df   = df.groupby(["MMSI"])
    
    tracks_mmsi = []
    tracks_Ts   = []
    tracks_XYs  = []
    for track in unique_mmsi:
        track_df    = tracks_df.get_group(track)
        track_df    = track_df.sort_values("BaseDateTime")
        track_dtgs  = pd.to_datetime(track_df["BaseDateTime"])                # dataframe with basedatetime
        track_times = track_dtgs.to_numpy()                                   # numpy array with datetime64 type
        print(type(track_times[0]))
        track_dT    = track_times[1:] - track_times[:-1]
        print("what the heckin", track_dT, type(track_dT))
        #track_times = track_times.astype(np.int64)
        #track_times = track_times.astype("datetime64[s]").astype('int')        # converts to unix seconds i think in
        #track_times = track_times.astype(np.float64)                          # convert to float 64s (to kind of prevent div by 0)
        track_XY    = track_df[["LON", "LAT"]].to_numpy()
        #track_TXY   = np.concatenate([np.expand_dims(track_times, axis=-1), track_pos], axis=-1)
        tracks_Ts.append(track_times)
        tracks_mmsi.append(track)
        tracks_XYs.append(track_XY)
    
    # could be done in one loop with the above
    # but this is slightly faster (don't get in the way of pd's groove i'd guess)
    # perhaps we can come back here and play with where np is used and pd is used
    tracks_maxspeeds = []
    for ndx, track_T in enumerate(tracks_Ts):
        track_XY  = tracks_XYs[ndx]
        dXY_dT    = __calc_vels(T=track_T, XY=track_XY)
        speeds    = np.linalg.norm(dXY_dT, axis=-1)
        max_speed = np.max(speeds)
        
        tracks_maxspeeds.append(max_speed)
    
    max_speed = max(tracks_maxspeeds)
    
    return (tracks_mmsi, tracks_Ts, tracks_XYs, max_speed)

# load the ais at path
# returns a list of track mmsi's, track (times, lons, lats), max speed for all tracks, and processing time
def load_ais_data(path: str) -> Tuple[list, list, float, int]:#, n_procs: int):
    tic = time.perf_counter()
    
    df  = pd.read_csv(path)
    tracks_mmsi, tracks_Ts, tracks_XYs, max_speed = __get_tracks(df=df)
    
    toc = time.perf_counter()
    
    return (tracks_mmsi, tracks_Ts, tracks_XYs, max_speed, toc - tic)
 
    
if __name__ == "__main__":

    path = "reduced_data/val.csv"
    data = load_ais_data(path=path)
    (tracks_mmsi, tracks_Ts, tracks_XYs, max_speed, proc_time) = data
    
    print("self reported load and process time was", proc_time)
        
    print("max of all speeds found to be", max_speed, "or", max_speed*60*60*60, "nautical miles per hour.  which is too damn fast")

