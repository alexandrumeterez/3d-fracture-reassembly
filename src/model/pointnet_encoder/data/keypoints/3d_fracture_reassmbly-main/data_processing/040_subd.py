import os
import copy
import numpy as np
import math as m 

from compas.datastructures import Mesh
from compas.datastructures import mesh_subdivide_corner
from compas.datastructures import mesh_explode

from compas_view2.app import App
from compas.utilities import i_to_rgb
# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
folder_name = 'cylinder_20_seed_9'
FILE_FOLDER = os.path.join(HERE, 'data', folder_name)

# ==============================================================================
# Analysis Data
# ==============================================================================

meshes = []
max_len_v = 2000

for i, filename in enumerate(os.listdir(FILE_FOLDER)):
    print(filename)
    if filename.endswith(".obj"):
        FILE_I = os.path.join(FILE_FOLDER, filename)
        if "shard" in filename: 
            mesh = Mesh.from_obj(FILE_I)
            exploded_meshes = mesh_explode(mesh)
            for ex_mesh in exploded_meshes:
                len_v = len(list(ex_mesh.vertices()))
                if len_v > 100:
                    meshes.append(ex_mesh)
                    if len_v > max_len_v:
                        max_len_v = len_v
        else:
                os.remove(FILE_I)

    elif filename.endswith(".mtl"):
        FILE_E = os.path.join(FILE_FOLDER, filename)
        os.remove(FILE_E)

print(max_len_v)

meshes_copy = copy.deepcopy(meshes)

# ==============================================================================
# Subdivide and Output
# ==============================================================================

print(len(meshes))
problem_ind = []
for i, mesh in enumerate(meshes):
    FILE_O = os.path.join(FILE_FOLDER, '%s_%s.npy' % (folder_name, i))

    len_f = len(list(mesh.faces()))
    len_v = len(list(mesh.vertices()))
    len_e = len(list(mesh.edges()))

    while len_v < max_len_v / 2:
        mesh = mesh_subdivide_corner(mesh, k=1)
        len_v = len(list(mesh.vertices()))

    print(max_len_v, len_v)

    try:
        vertices = np.array([mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()])
        normals = np.array([mesh.vertex_normal(vkey) for vkey in mesh.vertices()])
        datas = np.concatenate((vertices, normals), axis=1)
        np.save(FILE_O, datas)
    except:
        print("soemthing goes wrong...", i)
        problem_ind.append(i)

print(problem_ind)


# # ==============================================================================
# # Viz
# # ==============================================================================
# viewer = App()

# # for i, mesh in enumerate(meshes):
# #     viewer.add(mesh, facecolor=i_to_rgb(i/len(meshes), True))


# for i in problem_ind:
#     viewer.add(meshes_copy[i], facecolor=i_to_rgb(i/len(meshes_copy), True))

# viewer.run()





