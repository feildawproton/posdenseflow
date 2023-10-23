import taichi as ti

@ti.data_oriented
class MyGUI:
    @ti.kernel
    def __zero_3dvector2dfield(self, somefield: ti.template()):
        for i, j in somefield:
            somefield[i,j][0] = 0.0
            somefield[i,j][1] = 0.0
            somefield[i,j][2] = 0.0
            
    def __init__(self, x: int, y: int):
        self.width  = x
        self.height = y
        print("initializing to gui")
        self.gui = ti.GUI("hello fast gui", (self.width, self.height), fast_gui=True)
        n = 3
        self.scratch_vizfield = ti.Vector.field(n=3, dtype=ti.f32, shape=(x,y))
        self.__zero_3dvector2dfield(self.scratch_vizfield)
        print("we need a 3d vector form of our scalar for fast gui visualization")
        print("so we set up a scratch field for the gui to draw to inside MyGUI")
        print("it's shape is:", self.scratch_vizfield.shape, "with n:", n)
        
    @ti.kernel        
    def __write_scaler2scratchfield(self, scalar_field: ti.template()):
        for i, j in scalar_field:
            # don't like ifs but whatever
            if scalar_field[i,j] > 0:
                self.scratch_vizfield[i,j][0] = scalar_field[i,j]
                self.scratch_vizfield[i,j][1] = 0
                self.scratch_vizfield[i,j][2] = 0
            else:
                self.scratch_vizfield[i,j][0] = 0
                self.scratch_vizfield[i,j][1] = 0
                self.scratch_vizfield[i,j][2] = -scalar_field[i,j]
    
    @ti.kernel        
    def __write_vectos2scratchfield(self, vector_field: ti.template()):
        for i, j in vector_field:
            # don't like ifs but whatever
            x = vector_field[i,j][0]
            y = vector_field[i,j][1]
            z = vector_field[i,j][2]
            red    = max(x, 0)
            blue   = max(y, 0)
            green  = ti.abs(min(x, 0))
            yellow = ti.abs(min(y, 0))
            self.scratch_vizfield[i,j][0] = ti.abs(x)
            self.scratch_vizfield[i,j][1] = ti.abs(y)
            self.scratch_vizfield[i,j][2] = ti.abs(x*y)

        
    def set_1d(self, scalar_field: ti.template()):
        assert scalar_field.shape == self.scratch_vizfield.shape
        self.__write_scaler2scratchfield(scalar_field=scalar_field)
        if self.gui.running:
            #self.__update_posviz()
            self.gui.set_image(self.scratch_vizfield)
            
    def set_3d(self, vector_field: ti.template()):
        assert vector_field.shape[0] == self.width and vector_field.shape[1] == self.height
        self.__write_vectos2scratchfield(vector_field=vector_field)
        if self.gui.running:
            self.gui.set_image(self.scratch_vizfield)

    def display(self):
        self.gui.show()
 
        
        