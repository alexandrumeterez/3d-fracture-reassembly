import os
from compas.datastructures import Mesh
from compas.datastructures import mesh_transform_numpy
import compas.geometry as cg
from compas.utilities import i_to_rgb

from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_FOLDER = os.path.join(HERE, 'data', 'cube_20')

# initialize viewer
viewer = App()

file_nums = len(os.listdir(FILE_FOLDER))
print(file_nums)

for i, filename in enumerate(os.listdir(FILE_FOLDER)):
    FILE_I = os.path.join(FILE_FOLDER, filename)
    if filename.endswith(".obj") and "shard" not in filename:
        mesh = Mesh.from_obj(FILE_I)
        mass_center = mesh.centroid()

for i, filename in enumerate(os.listdir(FILE_FOLDER)):
    FILE_I = os.path.join(FILE_FOLDER, filename)
    if filename.endswith(".obj") and "shard" in filename:
        mesh = Mesh.from_obj(FILE_I)
        mesh_center = mesh.centroid()
        vec = cg.Vector(*[a - b for (a, b) in zip(mesh_center, mass_center)])
        vec = vec * 0.5
        T = cg.Translation.from_vector(vec)
        mesh_transform_numpy(mesh, T)
        viewer.add(mesh, facecolor=i_to_rgb(i/file_nums, True))

# ==============================================================================
# Viz
# ==============================================================================
viewer.run()
