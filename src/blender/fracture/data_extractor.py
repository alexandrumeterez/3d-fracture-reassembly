from ast import arg
import os
import numpy as np
from compas.datastructures import Mesh
from compas.datastructures import mesh_explode
from tqdm import tqdm 
import argparse

def extract_vertices_normals(input_dir, output_dir, threshold=100):
	files = os.listdir(input_dir)

	counter = 0
	for i, filename in enumerate(tqdm(files)):
		if not filename.endswith(".obj"):
			continue
			
		obj_file = os.path.join(input_dir, filename)
		mesh = Mesh.from_obj(obj_file)
		exploded_meshes = mesh_explode(mesh)
		for ex_mesh in exploded_meshes:
			if len(list(ex_mesh.vertices())) < threshold:
				continue
			vertices = np.array([ex_mesh.vertex_coordinates(vkey) for vkey in ex_mesh.vertices()])
			normals = np.array([ex_mesh.vertex_normal(vkey) for vkey in ex_mesh.vertices()])

			output_file = os.path.join(output_dir, f"{counter}.npy")
			datas = np.concatenate((vertices, normals), axis=1)
			np.save(output_file, datas)

			counter += 1
	print("Done")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--threshold', default=100, type=int)

	args = parser.parse_args()

	extract_vertices_normals(args.input_dir, args.output_dir, args.threshold)
