import os
import math as m
from compas.datastructures import Mesh,  mesh_connected_components, mesh_explode
from compas.datastructures import mesh_transform_numpy
from compas.geometry import matrix_from_axis_and_angle, Translation

from compas_view2.app import App
from compas.utilities import i_to_rgb

# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_FOLDER = os.path.join(HERE, 'data', 'cat_seed_3')
FILE_I = os.path.join(HERE, 'data', 'cat_seed_1/grp1.001_shard.001.obj')
FILE_O = os.path.join(HERE, 'data', 'cat_seed_1/grp1.001_shard.001.off')

# ==============================================================================
# Mesh
# ==============================================================================
mesh = Mesh.from_obj(FILE_I)
centorid = mesh.centroid()
T = Translation.from_vector([a * -1 for a in centorid])
mesh_transform_numpy(mesh, T)

# calculate the euler characteristic
# https://en.wikipedia.org/wiki/Euler_characteristic
print(mesh.euler())

# print(mesh.adjacency)
# connect_components = mesh_connected_components(mesh)

# TODO: fix this in blender output... 
# this obj file might contain several water-tight meshes...
# explode meshes to connected parts
exploded_meshes = mesh_explode(mesh)

# ==============================================================================
# Viz
# ==============================================================================
viewer = App()
for i, exploded_mesh in enumerate(exploded_meshes):
    viewer.add(exploded_mesh, facecolor=i_to_rgb(i/len(exploded_meshes), True))

# only show points
# viewer.add(mesh, show_points=True, show_lines=False, show_faces=False)
# show the mesh
viewer.run()