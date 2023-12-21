import taichi as ti
import os
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import sys

def draw_proc(width: int, height: int, fps_limit_py: int, draw_queue: mp.Queue):
    # -- PRINT INFORMATION --
    print("Launching a GLFW window with Taichi, from process:", os.getpid())
    
    # -- INIT VULKAN --
    # or not... ti.init(arch=ti.vulkan) # if not already
    
    # -- START WINDOW --
    # there's an fps limit inside the window implementation (default=1000).  but I want to sleep from within the python control logic
    # i don't want the internal glfw window to sleep when we could:
    # return control to us, do other stuff we need to, then sleep here in python
    window = ti.ui.Window(name="A GLFW window from taichi with proc id: "+str(os.getpid()), 
                          res=(width,height),
                          fps_limit=sys.maxsize,  # pretty much, just don't sleep internally  
                          pos=(500,300))
    
    # -- RUN --
    sec_per_frame = 1. / float(fps_limit_py)
    canvas = window.get_canvas()
    while window.running:
        t0 = time.perf_counter()         
           
        # -- BEGIN ALL WINDOW STUFF --
        while draw_queue.qsize() > 0:
            img = draw_queue.get()
            canvas.set_image(img=img)
            window.show()
        window.show() # second call to show at the end of the queue... hmmm
        # -- END ALL WINDOW STUFF --
        
        t1 = time.perf_counter()
        # -- WAIT TO LIMIT FPS AND FREE UP RESOURCES --
        dt     = t1 - t0
        t_left = sec_per_frame - dt
        t_pad  = 1./6_000. # because of other overhead beyond what is inside t1-t0. experimentally determined
        if t_left > t_pad:
            time.sleep(t_left - t_pad)

if __name__ == "__main__":
    print("Main function pid:", os.getpid())
    
    #ti.init(arch=ti.vulkan)
    
    width   = 640
    height  = 480
    max_fps = 60
    draw_queue = mp.Queue()
    gui_proc   = mp.Process(target=draw_proc, args=(width, height, max_fps, draw_queue))
    gui_proc.start()
    #ti.init(arch=ti.vulkan)
    ti.init(arch=ti.vulkan)
    viz_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))
    draw_queue.put(viz_field)
    gui_proc.join()
    
    print("Goodby main function with pid:", os.getpid())
    #window = ti.ui.Window(name="A GLFW window from taichi", res=(640,480), fps_limit=240, pos=(0,0))
    #canvas = window.get_canvas()
    # canvas.set_image(something)
    
    #while window.running:
    #    window.show()
        
    
    