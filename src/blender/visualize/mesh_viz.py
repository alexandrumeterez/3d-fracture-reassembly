import os
from compas.datastructures import Mesh
from compas.utilities import i_to_rgb
from compas_view2.app import App

class MeshViewer(object):
	def __init__(self):
		self.viewer = App()
	
	def visualize_mesh(self, path):
		mesh = Mesh.from_obj(path)
		len_vertices = len(list(mesh.vertices()))

		print(f'Number of vertices: {len_vertices}')
		self.viewer.add(mesh, facecolor=i_to_rgb(0.5, True))

if __name__ == '__main__':
	mesh_viewer = MeshViewer()
	pass