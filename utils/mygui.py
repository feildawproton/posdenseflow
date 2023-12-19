import taichi as ti
import time
import multiprocessing as mp

@ti.data_oriented
class MyGUI:
    @ti.kernel
    def __zero_3dvector2dfield(self, somefield: ti.template()):
        for i, j in somefield:
            somefield[i,j][0] = 0.0
            somefield[i,j][1] = 0.0
            somefield[i,j][2] = 0.0
            
    def __gui_go(self):
        while self.gui.running:
            self.gui.show()
            
    def __init__(self, width: int, height: int):
        print("setting up fast gui")
        tic = time.perf_counter()
        
        # -- INITIALIZE GUI --
        self.width  = width
        self.height = height
        self.gui    = ti.GUI("hello fast gui", (self.width, self.height), fast_gui=True)
        
        # -- SET UP THE FIELD THE GUI WILL DRAW FROM --
        n = 3
        self.viz_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(width,height))
        self.__zero_3dvector2dfield(self.viz_field)
        self.gui.set_image(self.viz_field)

        
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
            self.viz_field[i,j][1] = vector_field[i,j][0]
            self.viz_field[i,j][2] = vector_field[i,j][0]

        
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
 
        
        