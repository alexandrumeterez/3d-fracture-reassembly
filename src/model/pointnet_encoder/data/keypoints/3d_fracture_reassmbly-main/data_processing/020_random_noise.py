import os
import math as m
import random as r
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

# # get vertex key and coordinates
# for vkey in mesh.vertices():
#     xyz = mesh.vertex_coordinates(vkey)
#     print('vertex key: ', vkey, 'coordinate: ', xyz)

# ==============================================================================
# Add Noise to Mesh
# ==============================================================================
# max_noise_distance
max_dis_ratio = 0.05
# noise amount
amount_ratio = 0.05

bbox = cg.oriented_bounding_box_numpy([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
diag = cg.length_vector(cg.subtract_vectors(bbox[6], bbox[0]))
max_dis = max_dis_ratio * diag
# noise numbers 

amount = int(len(list(mesh.vertices())) * amount_ratio)

colors = {}
random_pts = set()

for n in range(amount):
    vkey = mesh.get_any_vertex()
    random_pts.add(vkey)
    normal = mesh.vertex_normal(vkey)
    length = r.uniform(-max_dis/3, max_dis)
    vec = cg.scale_vector(cg.normalize_vector(normal), length)
    xyz = mesh.vertex_coordinates(vkey)
    new_xyz = cg.add_vectors(vec, xyz)
    mesh.vertex_attributes(vkey, "xyz", new_xyz)

colors.update({vkey: (255, 0, 0) for vkey in list(random_pts)})

mesh.to_off(FILE_O, author="Chaoyu")
# ==============================================================================
# Viz
# ==============================================================================
viewer = App()
# only show points
viewer.add(mesh, show_points=True, show_lines=False, show_faces=False)
# show the mesh
# viewer.add(mesh_t)
viewer.run()