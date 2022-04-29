import os
import math as m
from compas.datastructures import Mesh
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg

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
# Load Mesh
# ==============================================================================
mesh = Mesh.from_obj(FILE_I)
centorid = mesh.centroid()
T = cg.Translation.from_vector([a * -1 for a in centorid])
mesh_transform_numpy(mesh, T)

# ==============================================================================
# Remesh
# ==============================================================================
print(mesh.is_trimesh())

# ==============================================================================
# Output
# ==============================================================================
mesh.to_off(FILE_O, author="Chaoyu")

# # ==============================================================================
# # Viz
# # ==============================================================================
# viewer = App()
# # only show points
# viewer.add(mesh_t, show_points=True, show_lines=False, show_faces=False)
# # show the mesh
# # viewer.add(mesh_t)
# viewer.run()