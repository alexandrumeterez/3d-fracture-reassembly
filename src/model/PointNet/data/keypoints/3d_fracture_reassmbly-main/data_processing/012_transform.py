import os
import math as m
from compas.datastructures import Mesh
from compas.datastructures import mesh_transform_numpy
from compas.geometry import matrix_from_axis_and_angle, Translation

from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_FOLDER = os.path.join(HERE, 'data', 'cat_seed_1')
file_name = "cat_seed_1/grp1.001_shard.004"
FILE_I = os.path.join(HERE, 'data', '%s.obj' % file_name)
FILE_O = os.path.join(HERE, 'data', '%s_transformed.off' % file_name)

# ==============================================================================
# Mesh
# ==============================================================================
mesh = Mesh.from_obj(FILE_I)
centorid = mesh.centroid()
T = Translation.from_vector([a * -1 for a in centorid])
mesh_transform_numpy(mesh, T)

# # get vertex key and coordinates
# for vkey in mesh.vertices():
#     xyz = mesh.vertex_coordinates(vkey)
#     print('vertex key: ', vkey, 'coordinate: ', xyz)

# transform mesh
T = matrix_from_axis_and_angle([0, 0.5, 0.3], m.pi / 4)
# create a copy of the original mesh 
mesh_t = mesh.copy()
mesh_transform_numpy(mesh_t, T)

mesh_t.to_off(FILE_O, author="Chaoyu")
# ==============================================================================
# Viz
# ==============================================================================
viewer = App()
# only show points
# viewer.add(mesh_t, show_points=True, show_lines=False, show_faces=False)
# show the mesh
viewer.add(mesh_t)
viewer.run()