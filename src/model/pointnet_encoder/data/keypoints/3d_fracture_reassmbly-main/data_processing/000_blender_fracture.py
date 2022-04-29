import os
import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatVectorProperty, IntVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add

from mathutils import Matrix
bpy.app.debug = True
bpyscene = bpy.context.scene


# primitive = "cube"
primitive = "cylinder"

for seed in range(1, 5):

    # delete all the meshes
    # bpy.ops.mesh.select_all(action='DESELECT')
    for o in bpyscene.objects:
        if o.type == 'MESH':
            o.select = True
        else:
            o.select = False
    bpy.ops.object.delete()

    # Create an empty mesh and the object.
    if primitive == "cube":
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0))

    # loop through all the objects in the scene
    for ob in bpyscene.objects:
        if ob.type == 'MESH':
            # make the current object active and select it
            bpyscene.objects.active = ob
            ob.select = True

    ob = bpy.context.active_object

    # change to edit mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT')
    print("Active object = ",ob.name)

    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)

    # subdivide
    bmesh.ops.subdivide_edges(bm,
                            edges=bm.edges,
                            cuts=40,
                            use_grid_fill=True,
                            )

    # Write back to the mesh
    bpy.ops.mesh.select_all(action='DESELECT')
    bm.select_flush(True)

    bpy.ops.object.mode_set(mode='OBJECT') # if bmesh.from_edit_mesh() --> mode == EDIT - ValueError: to_mesh(): Mesh 'Cube' is in editmode 
    bm.to_mesh(me) #If mode ==Object  -> ReferenceError: BMesh data of type BMesh has been removed
    bm.free() 
    ob.update_from_editmode()

    # variables
    count = 20

    # modifier = object.modifiers.new(name="Fracture", frac_algorithm='BOOLEAN_FRACTAL')
    bpy.ops.object.modifier_add(type='FRACTURE')
    md = ob.modifiers["Fracture"]
    md.fracture_mode = 'PREFRACTURED'
    md.frac_algorithm = 'BOOLEAN_FRACTAL'
    md.fractal_cuts = 2
    md.fractal_iterations = 4
    md.shard_count = count
    md.point_seed = seed  # random seed

    bpy.ops.object.fracture_refresh(reset=True)
    bpy.ops.object.rigidbody_convert_to_objects()



    folder = os.path.abspath("C:\\Users\\Chaoyu\\Documents\\Github\\3d_fracture_reassmbly\\data")
    name = primitive + "_" + str(count) + "_seed_" + str(seed)
    path = os.path.join(folder, name)
    os.makedirs(path, exist_ok=True)


    bpy.ops.object.select_all(action='DESELECT')    
    scene = bpy.context.scene
    for ob in scene.objects:
        scene.objects.active = ob
        ob.select = True

        if ob.type == 'MESH':
            bpy.ops.export_scene.obj(
                    filepath=os.path.join(path, ob.name + '.obj'),
                    use_selection=True,
                    )
        ob.select = False