import os
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from compas_view2.app import App
from compas.datastructures import Mesh, mesh_explode
from compas.datastructures import mesh_transform_numpy
from compas.geometry import Translation
import argparse

class MeshViewer(object):
	def __init__(self):
		self.viewer = App()

	def visualize_mesh(self, path, explode=False):
		mesh = Mesh.from_obj(path)

		if explode:
			centorid = mesh.centroid()
			T = Translation.from_vector([a * -1 for a in centorid])
			mesh_transform_numpy(mesh, T)
			exploded_meshes = mesh_explode(mesh)
			n_meshes = len(exploded_meshes)
			for i, exploded_mesh in enumerate(exploded_meshes):
				self.viewer.add(exploded_mesh, facecolor=i_to_rgb(i/n_meshes, True))
		else:
			self.viewer.add(mesh, facecolor=i_to_rgb(0.5, True))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mesh_path', type=str, required=True)
	parser.add_argument('--explode', action='store_true')

	args = parser.parse_args()

	mesh_viewer = MeshViewer()
	mesh_viewer.visualize_mesh(args.mesh_path, args.explode)