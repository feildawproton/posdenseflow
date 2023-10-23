import pandas as pd
import numpy as np
from typing import Tuple
import taichi as ti

from datafeeder import DataFeeder
from mygui import MyGUI

# if you wanted to njit some func
# from numba import njit
# and rewrite the cpu function appropriately outside of the class below

ONE_SIXTEENTH = 1 / 16.
ONE_HALF      = 1 / 2.
ONE_NINETH    = 1 / 9.

@ti.data_oriented
class Viewer:
    # csv_path is the path the the csv data to be opened in pandas format then converted
    # min_display_resolution is the resolution to apply to the smaller axis of lat or lon
    # the other axis will be greater and calculated relatively
    # time resolution is a string="Days", "Hours", "Minutes", "Seconds"
    @ti.kernel
    def __zero_2dscaler(self, somefield: ti.template()):
        for i, j in somefield:
            somefield[i,j] = 0.0
                
    # consider putting this in a utils files
    @ti.kernel
    def __zero_3dvector2dfield(self, somefield: ti.template()):
        for i, j in somefield:
            somefield[i,j][0] = 0.0
            somefield[i,j][1] = 0.0
            somefield[i,j][2] = 0.0
                
    def __init__(self, csv_path: str, min_field_res: int, time_res: str, decay_rate: float):
        ## -- DATA FEEDER -- ##
        print("initializing the feeder")
        self.feeder = DataFeeder(csv_path=csv_path, resolution=time_res)
        
        ## -- SIMULATION PARAMS -- ##
        self.lat_range  = self.feeder.max_lat - self.feeder.min_lat
        self.lon_range  = self.feeder.max_lon - self.feeder.min_lon
        print("latitude range is", self.lat_range, "from", self.feeder.min_lat, "to", self.feeder.max_lat)
        print("longitude range is", self.lon_range, "from", self.feeder.min_lon, "to", self.feeder.max_lon)
        self.t_n        = self.feeder.min_t
        print("starting time at", self.t_n, "with a resolution of", self.feeder.time_res)
        min_range       = min(self.lat_range, self.lon_range)
        pix_permin      = float(min_field_res) / min_range
        self.lat_pix    = int(self.lat_range * pix_permin)
        self.lat_pix    = max(self.lat_pix, min_field_res) # in case of rounding down
        self.lon_pix    = int(self.lon_range * pix_permin)
        self.lon_pix    = max(self.lon_pix, min_field_res) # for rounding
        print("there will be", self.lat_pix, "pixels for latitude and", self.lon_pix, "pixels for longitude")
        lat_perpix      = self.lat_range / float(self.lat_pix)
        lon_perpix      = self.lon_range / float(self.lon_pix)
        print("this turngs into", lat_perpix, "degrees of latitude per pixel and", 
              lon_perpix, "degrees of longitude per pixel")
        self.decay_rate = decay_rate
        print("decay rate set to", self.decay_rate, "with a time resolution of", self.feeder.time_res)
        
        ## -- TAICHI FIELDS -- ## 
        print("please make sure taichi is initialized")
        
        self.pos_field  = ti.field(dtype=ti.f32, shape=(self.lon_pix, self.lat_pix))
        self.__zero_2dscaler(self.pos_field)
        print("shape of pos_field", self.pos_field.shape, "lon and lat have flipped for easier viewing!!!")
        
        self.abs_pos    = ti.field(dtype=ti.f32, shape=(self.lon_pix, self.lat_pix))
        self.__zero_2dscaler(self.abs_pos)
        print("created an abs pos field for easier viz and such with shape", self.abs_pos.shape)
        
        n = 3
        self.pos_grad   = ti.Vector.field(n=n, dtype=ti.f32, shape=(self.lon_pix, self.lat_pix))
        self.__zero_3dvector2dfield(self.pos_grad)
        print("created a pos gradient vector field with shape", self.pos_grad.shape, "and n", n)
        
        self.last_grad  = ti.Vector.field(n=n, dtype=ti.f32, shape=(self.lon_pix, self.lat_pix))
        self.__zero_3dvector2dfield(self.last_grad)
        print("created a second pos gradient vector field with shape", self.last_grad.shape, "and n", n)
        
        self.time_grad  = ti.Vector.field(n=n, dtype=ti.f32, shape=(self.lon_pix, self.lat_pix))
        self.__zero_3dvector2dfield(self.time_grad)
        print("created a time-space gradient vector field with shape", self.time_grad.shape, "and n", n)
        
        ## -- TAICHI FAST GUI -- ##
        #self.gui = ti.GUI("hello fast gui", (self.lon_pix, self.lat_pix), fast_gui=True)
        print("gui implemented outside of here so that we can switch destimations")
        self.my_display = MyGUI(x=self.lon_pix, y=self.lat_pix)
                
    def __latlon2xy(self, lat: float, lon: float) -> Tuple[int, int]:
        lat_norm = (lat - self.feeder.min_lat) / self.lat_range
        lon_norm = (lon - self.feeder.min_lon) / self.lon_range
        x_pix    = int(lon_norm * self.lon_pix)
        y_pix    = int(lat_norm * self.lat_pix)
        #print(lon_norm, lat_norm, x_pix, y_pix)
        return (x_pix, y_pix) # don't forget to reverse order
        
    # this is serial to avoid a data race
    # doing it in python for simplicity
    def __add_samples2pos(self, samples: np.ndarray):
        assert len(samples.shape) == 2
        for sample in samples:
            x, y = self.__latlon2xy(lat=sample[0], lon=sample[1])
            #self.pos_field[x,y] += 1.0
            self.pos_field[x,y] = 1.0
            
    def __check__all_good(self):
        # just some asserts for sanity
        assert self.pos_field.shape[0] == self.lon_pix and self.pos_field.shape[1] == self.lat_pix # silly assert
        assert self.my_display.width == self.lon_pix and self.my_display.height == self.lat_pix
    
    @ti.kernel        
    def __blur_pos(self,):
        for i, j in self.pos_field:
            # if statements are lame in kernels
            # what's a better way to do this?  padding?  
            if i > 0 and j > 0 and i < (self.lon_pix - 1) and j < (self.lat_pix - 1):
                
                # gaussian blur
                b_00 = self.pos_field[i-1, j-1] * 1
                b_10 = self.pos_field[i  , j-1] * 2
                b_20 = self.pos_field[i+1, j-1] * 1
                b_01 = self.pos_field[i-1, j  ] * 2
                b_11 = self.pos_field[i  , j  ] * 4
                b_21 = self.pos_field[i+1, j  ] * 2
                b_02 = self.pos_field[i-1, j+1] * 1
                b_12 = self.pos_field[i  , j+1] * 2
                b_22 = self.pos_field[i+1, j+1] * 1
                blur = ONE_SIXTEENTH * (b_00 + b_10 + b_20 +
                                        b_01 + b_11 + b_21 + 
                                        b_02 + b_12 + b_22)
                #self.pos_field[i,j] = blur * self.decay_rate
                self.pos_field[i,j] = self.pos_field[i,j] - self.decay_rate * blur
                #signed = self.pos_field[i,j] - self.decay_rate * blur
                #self.pos_field[i,j] = ti.abs(signed)
                
                '''
                # avg blur blur
                b_00 = self.pos_field[i-1, j-1] 
                b_10 = self.pos_field[i  , j-1] 
                b_20 = self.pos_field[i+1, j-1] 
                b_01 = self.pos_field[i-1, j  ] 
                b_11 = self.pos_field[i  , j  ] * 2
                b_21 = self.pos_field[i+1, j  ] 
                b_02 = self.pos_field[i-1, j+1] 
                b_12 = self.pos_field[i  , j+1] 
                b_22 = self.pos_field[i+1, j+1] 
                blur = 0.1 * (b_00 + b_10 + b_20 +
                              b_01 + b_11 + b_21 + 
                              b_02 + b_12 + b_22)
                self.pos_field[i,j] = blur * self.decay_rate
                '''
                #self.pos_field[i,j] = self.pos_field[i,j] - self.decay_rate * blur
                #signed = self.pos_field[i,j] - self.decay_rate * blur
                #self.pos_field[i,j] = ti.abs(signed)

    @ti.kernel
    def __update_abspos(self):
        for i, j in self.pos_field:
            self.abs_pos[i,j] = ti.abs(self.pos_field[i,j])
            
    # takes an argument because i'm not sure we want to supply abs_pos or the wave pos
    # pos should be a scaler at any rate
    @ti.kernel
    def __calc_posgrad(self, pos: ti.template()):
        for i, j in pos:
            # if statements are lame in kernels
            # what's a better way to do this?  padding?  
            if i > 0 and j > 0 and i < (self.lon_pix - 1) and j < (self.lat_pix - 1):
                dx_0 = ((pos[i+1,j-1] - pos[i,j-1]) + (pos[i,j-1] - pos[i-1,j-1]))
                dx_1 = ((pos[i+1,j  ] - pos[i,j  ]) + (pos[i,j  ] - pos[i-1,j  ]))
                dx_2 = ((pos[i+1,j+1] - pos[i,j+1]) + (pos[i,j+1] - pos[i-1,j+1]))
                dx   = (dx_0 + 2*dx_1 + dx_2) #* 0.25
                dy_0 = ((pos[i-1,j+1] - pos[i-1,j]) + (pos[i-1,j] - pos[i-1,j-1]))
                dy_1 = ((pos[i  ,j+1] - pos[i  ,j]) + (pos[i  ,j] - pos[i  ,j-1]))
                dy_2 = ((pos[i+1,j+1] - pos[i+1,j]) + (pos[i+1,j] - pos[i+1,j-1]))
                dy   = (dy_0 + 2*dy_1 + dy_2) #* 0.25
                self.pos_grad[i,j][0] = dx_1
                self.pos_grad[i,j][1] = dy_1
                self.pos_grad[i,j][2] = pos[i,j]
                
    @ti.kernel
    def __calc_dxdt(self):
        for i, j in self.pos_grad:
            self.time_grad[i,j][0] = self.last_grad[i,j][0] - self.pos_grad[i,j][0]
            self.time_grad[i,j][1] = self.last_grad[i,j][1] - self.pos_grad[i,j][1]
            self.time_grad[i,j][2] = self.last_grad[i,j][2] - self.pos_grad[i,j][2]
            
    @ti.kernel
    def __reset_grad(self):
        for i, j in self.pos_grad:
            self.last_grad[i,j][0] = self.pos_grad[i,j][0]
            self.last_grad[i,j][1] = self.pos_grad[i,j][1]
            self.last_grad[i,j][2] = self.pos_grad[i,j][2]
        
    # doesn't have slicing so need to do this for 
    @ti.kernel
    def __zero_edges(self):
        for i, j in self.pos_field:
            if i == 0 or i == (self.pos_field.shape[0] - 1) or j == 0 or j == (self.pos_field.shape[1] - 1):
                self.pos_field[i,j] = 0.0
        
    def __increment(self):
        if self.t_n <= self.feeder.max_t:
            print("do some stuff")
            samples = self.feeder.get_samples(self.t_n)
            self.__add_samples2pos(samples)
            self.__check__all_good()  # remove later
            self.__blur_pos()
            self.__update_abspos()
            
            
            #self.__calc_posgrad(self.pos_field)
            self.__calc_posgrad(self.abs_pos)
            self.__calc_dxdt()
            self.__reset_grad()
            
            #self.my_display.set_1d(self.pos_field)
            #self.my_display.set_1d(self.abs_pos)
            self.my_display.set_3d(self.pos_grad)
            #self.my_display.set_3d(self.time_grad)
            
            self.my_display.display()
            
            self.__zero_edges()
            
            self.t_n += 1 #move time forward
        else:
            what = input("all out of data.  we should stop. press something")
            self.gui.running == False
            
    def run(self):
        while self.t_n <= self.feeder.max_t:
            self.__increment()         

if __name__ == "__main__":
    #run("jus_vabeach.csv")
    print("got this data from: https://marinecadastre.gov/ais/")
    ti.init(arch=ti.gpu) 
    viewer = Viewer(csv_path="jus_vabeach.csv", min_field_res=1080, time_res="minutes", decay_rate=0.02)
    viewer.run()