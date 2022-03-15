import os
import numpy as np

from compas.datastructures import Mesh
from compas.datastructures import mesh_explode

# ==============================================================================
# File
# ==============================================================================
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
folder_name = 'cylinder_20_seed_9'
FILE_FOLDER = os.path.join(HERE, 'data', folder_name)

# ==============================================================================
# Output
# ==============================================================================
counter = 0
for i, filename in enumerate(os.listdir(FILE_FOLDER)):
    if filename.endswith(".obj"):
        FILE_I = os.path.join(FILE_FOLDER, filename)
        if "shard" in filename: 
            mesh = Mesh.from_obj(FILE_I)

            # explode joined meshes
            exploded_meshes = mesh_explode(mesh)
            print(i, filename)
            for ex_mesh in exploded_meshes:
                FILE_O = os.path.join(FILE_FOLDER, '%s_%s.npy' % (folder_name, counter))
                # delete tiny pieces
                if len(list(ex_mesh.vertices())) < 100:
                    continue
                
                vertices = np.array([ex_mesh.vertex_coordinates(vkey) for vkey in ex_mesh.vertices()])
                normals = np.array([ex_mesh.vertex_normal(vkey) for vkey in ex_mesh.vertices()])

                datas = np.concatenate((vertices, normals), axis=1)
                print(np.shape(datas))
                np.save(FILE_O, datas)

                counter += 1

        else:
            os.remove(FILE_I)

    elif filename.endswith(".mtl"):
        FILE_E = os.path.join(FILE_FOLDER, filename)
        os.remove(FILE_E)





