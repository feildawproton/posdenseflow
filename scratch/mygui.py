import taichi as ti
import time
import multiprocessing as mp
import os

@ti.data_oriented
class MyGUI:
    @ti.kernel
    def __zero_vizfield(self):
        for i, j in self.viz_field:
            self.viz_field[i,j][0] = 0.0
            self.viz_field[i,j][1] = 0.0
            self.viz_field[i,j][2] = 0.0
            
    def __draw_loop(self):
        while self.gui.running:
            self.gui.show()
            
    def __init__(self, width: int, height: int):
        print("setting up fast gui")
        tic = time.perf_counter()
        
        # -- INIT TI IF NOT ALREADY --
        ti.init(arch=ti.vulkan)
        
        # -- INITIALIZE GUI --
        self.width  = width
        self.height = height
        self.gui    = ti.GUI("hello fast gui", (self.width, self.height), fast_gui=True)
        
        # -- SET UP THE FIELD THE GUI WILL DRAW FROM --
        self.viz_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.width,self.height))
        self.__zero_vizfield()
        self.gui.set_image(self.viz_field)
        
        # -- QUEUE FOR DRAW COMMANDS --
        draw_queue = mp.Queue()
        print("Starting GUI Process")
        draw_proc  = mp.Process(target=self.__draw_loop, args = ())
        draw_proc.start()
        draw_proc.join()
        

    '''
    @ti.kernel        
    def __copy_scalarfield_tovizfield(self, scalar_field: ti.template()):
        for i, j in scalar_field:
            self.viz_field[i,j][0] = ti.abs(scalar_field[i,j])
            self.viz_field[i,j][1] = ti.abs(scalar_field[i,j])
            self.viz_field[i,j][2] = ti.abs(scalar_field[i,j])

    @ti.kernel        
    def __copy_vectorfield_tovizfield(self, vector_field: ti.template()):
        for i, j in vector_field:
            self.viz_field[i,j][0] = vector_field[i,j][0]
            self.viz_field[i,j][1] = vector_field[i,j][1]
            self.viz_field[i,j][2] = vector_field[i,j][2]

        
    def set_scalarfield_viz(self, scalar_field: ti.template()):
        assert scalar_field.shape[0] == self.width and scalar_field.shape[1] == self.height
        self.__copy_scalarfield_tovizfield(scalar_field=scalar_field)
        if self.gui.running:
            self.gui.set_image(self.viz_field)
            
    def set_vectorfield_viz(self, vector_field: ti.template()):
        assert vector_field.shape[0] == self.width and vector_field.shape[1] == self.height
        self.__copy_vectorfield_tovizfield(vector_field=vector_field)
        if self.gui.running:
            self.gui.set_image(self.viz_field)

    def display(self):
        self.gui.show()
    '''
    
def gui_proc(width: int, height: int, draw_queue: mp.Queue):
    # -- PRINT INFO --
    print("Starting GUI from process:", os.getpid())
    print("Under current testing, ti gui.show() must sleep internally to ensure 60 fps. it must record the last time it was called, right?")
    
    # -- INIT TI IF NOT ALREADY --
    ti.init(arch=ti.vulkan)
    
    # -- INITIALIZE GUI --
    gui = ti.GUI("hello fast gui", (width, height), fast_gui=True)
    print("blah blah blah")
    
    # -- RUN -- 
    runs_through = 0
    t0 = time.perf_counter()
    while gui.running == True:
        if gui.get_event(ti.GUI.ESCAPE):
            break
        while draw_queue.qsize() > 0:
            gui.set_image(draw_queue.get())
        gui.show() # does gui.show() sleep? it seems to stick to 60 fps
        runs_through += 1
    t1 = time.perf_counter()
    print("ran through gui loop", runs_through, "times")
    print("that took", t1 - t0, "seconds")
    print("average fps:", runs_through/(t1-t0))
    
if __name__ == "__main__":
    
    print("Main function pid:", os.getpid())
    
    draw_queue = mp.Queue()
    gui_proc   = mp.Process(target=gui_proc, args=(640, 480, draw_queue))
    gui_proc.start()
    gui_proc.join()
    
    print("Goodby main function with pid:", os.getpid())
    '''
    import numpy as np
    ti.init(arch=ti.vulkan)
    begins = np.random.random((100, 2))
    directions = np.random.uniform(low=-0.05, high=0.05, size=(100, 2))
    gui = ti.GUI('arrows', res=(400, 400))
    while gui.running:
        gui.arrows(orig=begins, direction=directions, radius=1)
        gui.show()
    '''
    
 
        
        