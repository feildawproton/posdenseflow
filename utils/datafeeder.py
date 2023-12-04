import pandas as pd
import numpy as np
from typing import Tuple

class DataFeeder():
    # takes a csv 
    # sorts
    # forms a latlon array and time array
    # 2 array so that we can use floats in latlon and ints in time
    def __init__(self, csv_path: str, resolution="seconds"):
        print("getting data", csv_path)
        df     = pd.read_csv(csv_path)
        print("sorting by time")
        df.sort_values(by=["Year", "Month", "Day", "Hour", "Minute", "Second"], inplace=True)
        
        times   = df[["Year", "Month", "Day", "Hour", "Minute", "Second"]].to_numpy()
        print("shape of times", times.shape)
        day     = times[:,2]
        hour    = times[:,3]
        minute  = times[:,4]
        second  = times[:,5]
        
        days    = day
        hours   = np.add(hour, np.multiply(24, days))
        minutes = np.add(minute, np.multiply(60, hours))
        seconds = np.add(second, np.multiply(60, minutes))
        
        if resolution == "days":
            self.time     = days
            self.time_res = "days"
        elif resolution == "hours":
            self.time     = hours
            self.time_res = "hours"
        elif resolution == "minutes":
            self.time     = minutes
            self.time_res = "minutes"
        else:
            self.time = seconds
            self.time_res = "seconds"
        print("shape of time array", self.time.shape)
        print("setting the resoltuion to", self.time_res)
        print("time does not account for years and months, so will reset every month")
        
        self.latlon    = df[["LAT", "LON"]].to_numpy()
        print("shape of latlon array", self.latlon.shape)
        
        self.N_samples = self.latlon.shape[0]
        print(self.latlon.shape[0])
        print(self.time.shape[0])
        assert self.N_samples == self.time.shape[0]
        print("there are", self.N_samples, "samples in this dataset")
        #self.n_sample  = 0
        #print("setting the current sample to", self.n_sample)
        
        
        self.max_t     = np.max(self.time)
        self.min_t     = np.min(self.time)
        self.min_lat   = np.min(self.latlon[:,0])
        self.max_lat   = np.max(self.latlon[:,0])
        self.min_lon   = np.min(self.latlon[:,1])
        self.max_lon   = np.max(self.latlon[:,1])
    
    # returns a tuple of a time and a latlon
    '''
    def get_sample(self, i: int) -> Tuple[int, np.ndarray]:
        t_i      = self.time[i]
        latlon_i = self.latlon[i]
        return (t_i, latlon_i)
    '''
    
    # put in a 
    
    def get_samples(self, t: int) -> np.ndarray:
        ndc = np.where(self.time==t)[0]
        #print(ndc)
        print(ndc.shape[0], "samples found at time:", t, "with res", self.time_res)
        samples = self.latlon[ndc]
        print("returning an array with shape", samples.shape)
        return samples
        
        
    
        
    
        
        