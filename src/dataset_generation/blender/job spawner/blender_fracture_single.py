import os, math, sys
import bpy
import bmesh

bpy.app.debug = True
bpyscene = bpy.context.scene

"""
Blender Auto Fracture Script
---------------------------------
Prerequisites: Blender 2.79 Fracture Build

This script does:
    -import obj or stl meshes (uncomment the according line on line 46/47)
    -place their center at 0,0
    -remesh and subdivide them to approx 50k vertices
    -fracture them with settings below
    -save them to obj files

Instructions:
    -Run this script with Blender from the command line
         and pass the arguments for shard count, seed count and the model file name
     Usage: ./blender --background --python blender_fracture_single.py -- <shard_count> <seed_count> <model_name> [<model_name>]
     (a list of object names can be supplied as well)
     Example: ./blender --background --python blender_fracture_single.py -- 10 1 model.obj  
                will fragment the model 'model.obj' into 10 shards with a single seed
    or on a Cluster:
        bsub -R "rusage[mem=8000]" ./blender --background --python blender_fracture_single.py -- 1 10 model.obj
    -Outputs will be saved to the Blender Folder/script-output 
"""

# variables
input_path = os.path.abspath("script-input")

bpy.ops.object.select_all(action='SELECT') 
bpy.ops.object.delete()   

# get commandline arguments
argv = sys.argv
index = argv.index("--") + 1
shard_count = int(argv[index])
seed_count = int(argv[index+1])
model_list = argv[index+2:]
print("Splitting objects: ", model_list, " with ", shard_count, " shards")
for o in model_list:
    # bpy.ops.import_mesh.stl(filepath=os.path.join(input_path, o))
    bpy.ops.import_scene.obj(filepath=os.path.join(input_path, o))
    bpy.context.scene.objects.active = bpy.context.selected_objects[0]
    # replace name of obj with filename
    obj = bpy.context.active_object
    obj.name = o[:-4]
    obj.data.name = o[:-4]
    
    #remesh
    print("Remesh Object "+obj.name)
    bpy.ops.object.modifier_add(type='REMESH')
    obj.modifiers['Remesh'].mode = 'SMOOTH'
    obj.modifiers['Remesh'].octree_depth = 8 # too low value can cause non watertight
    bpy.ops.object.modifier_apply(modifier='Remesh',apply_as='DATA')
    
    ## Edit Mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_non_manifold(extend=False)
    #dissolve faces
    bpy.ops.mesh.dissolve_verts()
    bpy.ops.object.mode_set(mode='OBJECT')

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

    print("Imported Object: ", o)

    obj = obj
    # fracture objects
    if obj.type == 'MESH':
        current_mesh = obj.name
        print("Processing ", current_mesh)
        bpy.context.scene.objects.active = obj
        obj.select = True
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        obj.location.x = 0
        obj.location.y = 0
        obj.location.z = 0
    else:
        continue
    
    for seed in range(seed_count):
        bpy.context.scene.objects.active = obj
        obj.select = True
        ob = bpy.context.active_object
        
        bpy.ops.object.modifier_add(type='FRACTURE')
        md = ob.modifiers["Fracture"]
        md.fracture_mode = 'PREFRACTURED'
        md.frac_algorithm = 'BOOLEAN_FRACTAL'
        md.fractal_amount = 0.3 #0.3 for realistic fracture surface roughness
        md.fractal_cuts = 3
        md.fractal_iterations = 4
        md.shard_count = shard_count
        md.point_seed = seed  # random seed

        print("Fracture Object", ob.name)
        bpy.ops.object.fracture_refresh(reset=True)
        bpy.ops.object.rigidbody_convert_to_objects()

        bpy.ops.object.select_all(action='DESELECT')    
        scene = bpy.context.scene
        
        folder = os.path.abspath("script-output")
        if len(scene.objects) < shard_count + 1:
            print("Not enough shards, drop object:" + current_mesh)
            with open(os.path.join(folder,"skipped_"+current_mesh+".txt"), "a") as f:
                f.write(current_mesh + " #Fragments: " + str(len(scene.objects)-1))
            break

        for ob in list(scene.objects):
            if ob == obj or ob.type != 'MESH':
                continue
            scene.objects.active = ob
            ob.select = True
            bpy.ops.mesh.separate(type='LOOSE')
            ob.select = False
        
        bpy.ops.object.select_all(action='DESELECT')    
        scene = bpy.context.scene
        

        name = current_mesh + "_" + str(shard_count) + "_seed_" + str(seed)
        path = os.path.join(folder, name)
        os.makedirs(path, exist_ok=True)

        frag_num = 0
        for ob in list(scene.objects):
            scene.objects.active = ob
            ob.select = True
            
            if ob.type == 'MESH' and current_mesh in ob.name and current_mesh != ob.name:
                if len(ob.data.vertices)>1000:
                    bpy.ops.export_scene.obj(
                            filepath=os.path.join(path, current_mesh + "_shard_" + str(frag_num)+ '.obj'),
                            use_normals=True,
                            use_uvs=False, 
                            use_materials=False,
                            use_selection=True,
                            use_triangles=True
                            )
                    frag_num += 1
                    print("Save object", ob.name)
                else:
                    print("Discard Fragment"+ob.name)
                bpy.ops.object.delete()
    bpy.ops.object.select_all(action='SELECT') 
    bpy.ops.object.delete() 

print("DONE")
