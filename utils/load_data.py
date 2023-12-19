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
'''

# returns a dictionary
# where the keys are the track's mmsi
# and the values are a float 64 numpy matrix with columns: time, lon, lat, sog
# time is in seconds
# lon and lat are in degrees
# sog is in nm/hr
def __get_tracks(df: pd.DataFrame) -> dict:
    unique_mmsi = df.MMSI.unique()
    tracks_df   = df.groupby(["MMSI"])

    tracks_dict = {}
    for track in unique_mmsi:
        track_df    = tracks_df.get_group(track)
        track_df    = track_df.sort_values("BaseDateTime")
        
        # -- GET TIMES --
        # verbose with more arrays than necessary b.c. confusing 
        track_dtgs  = pd.to_datetime(track_df["BaseDateTime"])     # dataframe with basedatetime
        track_times = track_dtgs.to_numpy()                        # numpy array with datetime64 type (in nano seconds)
        track_t_f64 = track_times.astype(np.float64)               # in nanoseconds by now 64 bit floats
        track_t_sec = np.multiply(1. / 1_000_000_000, track_t_f64) # now in seconds with 64 bit floats
        
        # -- GET LON, LAT, and Speed Over Ground --
        track_XYS = track_df[["LON", "LAT", "SOG"]].to_numpy()     # in float 64s, units are degrees lon, lat and nm/hr for sog

        # -- ASSEMBLE MATRIX --
        track_len        = track_t_sec.shape[0]
        assert track_len == track_XYS.shape[0]
        track_TXYS       = np.zeros(shape=(track_len, 4))
        track_TXYS[:,0]  = track_t_sec
        track_TXYS[:,1:] = track_XYS
        
        # -- ADD TRACKS TO LIST --
        tracks_dict[track] = track_TXYS
        
    return tracks_dict

# load the ais at path
# returns a list of track mmsi's, track (times, lons, lats), max speed for all tracks, and processing time
def get_ais_frompath(path: str) -> Tuple[list, list, float, int]:#, n_procs: int):
    print("loading ais from path", path, "and processing into tracks")
    tic     = time.perf_counter()
    
    df          = pd.read_csv(path)
    tracks_dict = __get_tracks(df=df)
    
    toc     = time.perf_counter()
    tic2toc = toc - tic
    
    print("loading ais from path", path, "and processing into float 64s took:", tic2toc)
    
    return tracks_dict
 
    
if __name__ == "__main__":

    path      = "../reduced_data/train.csv"
    data_dict = get_ais_frompath(path=path)

