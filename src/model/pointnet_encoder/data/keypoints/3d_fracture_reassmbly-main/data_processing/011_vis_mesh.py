import os
import math as m
from compas.datastructures import Mesh,  mesh_connected_components, mesh_explode
from compas.datastructures import mesh_transform_numpy
from compas.geometry import matrix_from_axis_and_angle, Translation

from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_FOLDER = os.path.join(HERE, 'data', 'cat_seed_1')
FILE_I = os.path.join(HERE, 'data', 'cat_seed_1/grp1.001_shard.001.obj')
FILE_O = os.path.join(HERE, 'data', 'cat_seed_1/grp1.001_shard.001.off')

# ==============================================================================
# Mesh
# ==============================================================================
mesh = Mesh.from_obj(FILE_I)
centorid = mesh.centroid()
T = Translation.from_vector([a * -1 for a in centorid])
mesh_transform_numpy(mesh, T)

mesh.to_off(FILE_O, author="Chaoyu")

# ==============================================================================
# Viz
# ==============================================================================
viewer = App()
# only show points
# viewer.add(mesh, show_points=True, show_lines=False, show_faces=False)
# show the mesh
viewer.add(mesh)
viewer.run()