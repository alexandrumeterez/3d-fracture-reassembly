import bpy
bl_info = {    "name": "Shape to MESH",
    "author": "Albertofx",
    "version": (1, 0),
    "blender": (2,7,1),
        "location": "View3D > ToolShelf",
    "description": "Sets shape to MESH",
    "warning": "",
    "wiki_url": "",
    "category": "Mesh"}

def sm(self):
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.use_deform = True
    bpy.context.object.rigid_body.collision_margin = 0

class shapemeshoperator(bpy.types.Operator):
    """shape to mesh"""
    bl_idname = "mesh.shapemesh"
    bl_label = "Run"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        sm(self)
        return {'FINISHED'}

class shapemeshPanel(bpy.types.Panel):
    bl_category = "Fracture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    #bl_context = "editmode"
    bl_label = "Shape to MESH"
    
    
    def draw(self, context):
        
        layout = self.layout
        row = layout.row(align=True)
        layout.label("Will only work when object is fractured.", icon='INFO')
        
        row.operator(shapemeshoperator.bl_idname)
    
    def draw(self, context):
        if not context.object:
            self.layout.label("Please select atleast one object first")
        else:
            layout = self.layout
            layout.label("Decreases Fractal Explosion.")
            row = layout.row(align=True)
            
            row.operator(shapemeshoperator.bl_idname)
            layout.label("Run only after object has been fractured.", icon='INFO')

def register():
    bpy.utils.register_module(__name__)

def unregister():
    bpy.utils.unregister_module(__name__)

if __name__ == "__main__":
    register()