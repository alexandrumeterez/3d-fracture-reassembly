import os, math
import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatVectorProperty, IntVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Matrix
bpy.app.debug = True
bpyscene = bpy.context.scene

"""
Blender Auto Fracture Script
---------------------------------
Prerequisites: Blender 2.79 Fracture Build

This script does:
    -import obj meshes
    -place their center at 0,0
    -subdivide them to approx 50k vertices
    -fracture them with settings below
    -save them to ply files

Instructions:
    -Add all Meshes to the folder /script-input in the folder where blender is located
    -Run this script with Blender, i.e. blender --background --python myscript.py
    -Outputs will be saved to the Blender Folder/script-output
    
Note that Blender will freeze during execution. You can see what it's doing in the terminal.
    (Enable with: Window > Toggle System Console)
"""

# variables
shard_count = 8
seed_count = 4


bpy.ops.object.select_all(action='SELECT') 
bpy.ops.object.delete()   

# delete scene and add all objects from /script_input
input_path = os.path.abspath("script-input")
for o in os.listdir(input_path):
    bpy.ops.import_scene.obj(filepath=os.path.join(input_path, o))
    # replace name of obj with filename
    for obj in bpy.context.selected_objects:
            obj.name = o[:-4]
            obj.data.name = o[:-4]
            
            #subdivide to at somewhere near 42k points
            if(len(obj.data.vertices) < 42000):
                me = obj.data
                # New bmesh
                bm = bmesh.new()
                # load the mesh
                bm.from_mesh(me)

                # subdivide
                cut = int(math.sqrt(42000/len(obj.data.polygons)))+1
                bmesh.ops.subdivide_edges(bm,
                                          edges=bm.edges,
                                          cuts=cut,
                                          use_grid_fill=True,
                                          )

                # Write back to the mesh
                bm.to_mesh(me)
                me.update()
            bpy.ops.object.select_all(action='DESELECT')   
            
    print("Imported Object: ", o)

# fracture objects
object_list = list(bpyscene.objects)
for object in object_list:
    if object.type == 'MESH':
        current_mesh = object.name
        print("Processing ", current_mesh)
        bpy.context.scene.objects.active = object
        object.select = True
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        object.location.x = 0
        object.location.y = 0
        object.location.z = 0
    else:
        continue
    
    for seed in range(seed_count):
        bpy.context.scene.objects.active = object
        object.select = True
        ob = bpy.context.active_object
        
        bpy.ops.object.modifier_add(type='FRACTURE')
        md = ob.modifiers["Fracture"]
        md.fracture_mode = 'PREFRACTURED'
        md.frac_algorithm = 'BOOLEAN_FRACTAL'
        md.fractal_cuts = 3
        md.fractal_iterations = 4
        md.shard_count = shard_count
        md.point_seed = seed  # random seed

        print("Fracture Object", ob.name)
        bpy.ops.object.fracture_refresh(reset=True)
        bpy.ops.object.rigidbody_convert_to_objects()


        folder = os.path.abspath("script-output")
        name = current_mesh + "_" + str(shard_count) + "_seed_" + str(seed)
        path = os.path.join(bpy.app.binary_path, folder, name)
        os.makedirs(path, exist_ok=True)


        bpy.ops.object.select_all(action='DESELECT')    
        scene = bpy.context.scene
        
        for ob in list(scene.objects):
            if ob in object_list or ob.type != 'MESH':
                continue
            scene.objects.active = ob
            ob.select = True
            bpy.ops.mesh.separate(type='LOOSE')
            ob.select = False
        
        bpy.ops.object.select_all(action='DESELECT')    
        scene = bpy.context.scene
        
        frag_num = 0
        for ob in list(scene.objects):
            scene.objects.active = ob
            ob.select = True
            
            if ob.type == 'MESH' and current_mesh in ob.name and current_mesh != ob.name:
                if len(ob.data.vertices)>1000:
                    bpy.ops.export_mesh.ply(
                            filepath=os.path.join(path, current_mesh + "_shard_" + str(frag_num)+ '.ply'),
                            use_normals=True,
                            use_uv_coords=False, 
                            use_colors=False
                            )
                    frag_num += 1
                    print("Save object", ob.name)
                else:
                    print("Discard Fragment"+ob.name)
                bpy.ops.object.delete()

print("DONE")