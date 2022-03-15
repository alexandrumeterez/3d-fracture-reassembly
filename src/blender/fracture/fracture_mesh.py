import os
import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatVectorProperty, IntVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Matrix

def LOG(text):
	print(f'[INFO] {text}')

class Scene(object):
	def __init__(self):
		# Get scene object
		self.bpyscene = bpy.context.scene

	def select_all_meshes(self):
		scene_objects = bpyscene.objects
		# Iterate through every object in the scene and select it
		# if it's a mesh
		for scene_object in scene_objects:
			scene_object.select = (scene_object.type == 'MESH')
		LOG("All meshes selected")

	def add_empty_mesh(self, name, location):
		if name == 'cube':
			bpy.ops.mesh.primitive_cube_add(location=location)
		elif name == 'cylinder':
			bpy.ops.mesh.primitive_cylinder_add(location=location)
		LOG(f"Added mesh {name} at {location}")
	
	def set_mode(self, mode):
		if bpy.ops.object.mode_set.poll():
			bpy.ops.object.mode_set(mode=mode)
		LOG(f"Mode set to {mode}")
	
	def get_and_fracture_object(self, cuts=40, shard_count=20, seed=0):
		scene_objects = bpyscene.objects

		# Set as active the mesh in the scene
		# Assumes only one MESH in the scene
		for scene_object in scene_objects:
			if scene_object.type == 'MESH':
				scene_objects.active = scene_object
				scene_object.select = True
		
		active_scene_object = bpy.context.active_object
		LOG(f"Active object: {active_scene_object.name}")
		self.set_mode('EDIT')

		# Create mesh bmesh object from the selected mesh object
		mesh = bmesh.new().from_mesh(active_scene_object.data)

		# Subdivide the mesh into chunks
		bmesh.ops.subdivide_edges(mesh, edges=mesh.edges, cuts=cuts, use_grid_fill=True)

		# Deselect all objects
		bpy.ops.mesh.select_all(action='DESELECT')
		mesh.select_flush(True)

		# Set object mode
		self.set_mode('OBJECT')

		# Save broken mesh into the data of the active scene object
		mesh.to_mesh(active_scene_object.data)
		mesh.free() 
		active_scene_object.update_from_editmode()

		# Apply modifiers (I assume to make the cut more realistic)
		bpy.ops.object.modifier_add(type='FRACTURE')
		md = active_scene_object.modifiers["Fracture"]
		md.fracture_mode = 'PREFRACTURED'
		md.frac_algorithm = 'BOOLEAN_FRACTAL'
		md.fractal_cuts = 2
		md.fractal_iterations = 4
		md.shard_count = shard_count
		md.point_seed = seed  # random seed

		# This does some magic
		bpy.ops.object.fracture_refresh(reset=True)
		bpy.ops.object.rigidbody_convert_to_objects()

		bpy.ops.object.select_all(action='DESELECT')
	
	def save_all(self, path):
		scene_objects = bpyscene.objects
		for scene_object in scene_objects:
			scene_objects.active = scene_object
			scene_object.select = True

			if scene_object.type == 'MESH':
				bpy.ops.export_scene.obj(filepath=os.path.join(path, scene_object.name + '.obj'), use_selection=True)
			scene_object.select = False


