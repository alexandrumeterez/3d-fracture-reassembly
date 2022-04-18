import numpy as np
from scipy import spatial
import os
import argparse
import time
import scipy
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots 
from scipy.optimize import curve_fit
from multiprocessing import Pool

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

def objective(X, a0, a1, a2, a3, a4, a5):
	x = X[:, 0]
	y = X[:, 1]
	return a0 * (x ** 2) + a1 * (y ** 2) + a2 * (x * y) + a3 * x + a4 * y + a5

class Extractor(object):
	def __init__(self, fragment_path, output_path, keypoint_radius, r_values, n_keypoints):
		self.fragment_path = fragment_path
		self.output_path = output_path
		self.r_vals = r_values
		self.keypoint_radius = keypoint_radius
		self.n_keypoints = n_keypoints
		self.cov_mats = []

	def compute_SD_point(self, neighbourhood, points, normals, p_idx):
		p_i = points[p_idx]
		n_p_i = normals[p_idx]
		p_i_bar = np.mean(points[neighbourhood], axis=0)
		v = p_i - p_i_bar
		SD = np.dot(v, n_p_i)
		return SD

	# Assembling the above
	def get_SD_for_point_cloud(self, point_cloud, normals, neighbourhood):
		n_points = len(point_cloud)
		# Compute SD
		SD = np.zeros((n_points))
		for i in range(n_points):
			SD[i] = self.compute_SD_point(np.asarray(neighbourhood[i]), point_cloud, normals, i)
		return SD

	def extract(self):
		fragment = np.load(self.fragment_path)
		point_cloud = fragment[:, :3]
		normals = fragment[:, 3:]
		self.point_cloud = point_cloud
		point_cloud = np.asarray(point_cloud, order='F')
		normals = np.asarray(normals, order='F')

		# Get all radius r neighbourhoods for each r
		keypoint_radius = self.keypoint_radius
		tree = spatial.KDTree(point_cloud)

		# Extract keypoints
		nbhd = tree.query_ball_point(point_cloud, keypoint_radius, workers=-1)
		SD = self.get_SD_for_point_cloud(point_cloud, normals, nbhd)

		# Extract keypoint indices
		keypoint_indices = np.argsort(np.abs(SD))[-self.n_keypoints:]
		self.keypoints = point_cloud[keypoint_indices]
		# Compute the neighbourhoods in all r vals 
		neighbourhoods = {}
		for r in self.r_vals:
			neighbourhoods[r] = tree.query_ball_point(point_cloud, r, workers=-1)

		# Output
		output_matrix = np.zeros((len(keypoint_indices), 3 + 3 + 49 * len(self.r_vals)))

		# For each keypoint
		for n_keypoint, keypoint_index in enumerate(keypoint_indices):
			# Get keypoint and normal of the keypoint
			keypoint = point_cloud[keypoint_index]
			keypoint_normal = normals[keypoint_index]

			# Set output matrix
			output_matrix[n_keypoint, :3] = keypoint
			output_matrix[n_keypoint, 3:6] = keypoint_normal

			# For each radius r, compute the matrix with the features
			for r_idx, r in enumerate(self.r_vals):
				# Get neighbourhood of keypoint
				keypoint_neighbourhood = neighbourhoods[r][keypoint_index]

				# Initialize Phi
				Phi = []

				for p_i_idx in keypoint_neighbourhood:
					p_i = point_cloud[p_i_idx]
					v = p_i - keypoint

					if np.linalg.norm(v) < 1e-5:
						continue
					
					cos_alpha = np.dot(v, normals[p_i_idx])
					cos_alpha /= np.linalg.norm(v)

					cos_beta = np.dot(v, keypoint_normal)
					cos_beta /= np.linalg.norm(v)

					cos_gamma = np.dot(normals[p_i_idx], keypoint_normal)

					phi_i = [cos_alpha, cos_beta, cos_gamma]
					Phi.append(phi_i)

				Phi = np.asarray(Phi)
				Phi = Phi.T
				cov_mat = np.cov(Phi)

				S, R = np.linalg.eig(cov_mat)
				S = np.log(S)
				cov_mat = R @ np.diag(S) @ R.T

				self.cov_mats.append(cov_mat)

		np.save(self.output_path, output_matrix)
def f(fragment_path, output_path, keypoint_radius, r_vals, n_keypoints):
	extractor = Extractor(fragment_path, output_path, keypoint_radius, r_vals, n_keypoints)
	extractor.extract()

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--dataset_dir", type=str)
	# parser.add_argument("--keypoint_radius", type=float, required=True)
	# parser.add_argument("--n_keypoints", type=int, required=True)
	# parser.add_argument("--r_vals", nargs='+', required=True, type=float)

	# args = parser.parse_args()
	n_points = 2000
	extractor1 = Extractor("/Users/alex/3dv_mockup/Venus/venus_part01.npy", ".", 0.1, [0.2], n_points)
	extractor1.extract()
	extractor2 = Extractor("/Users/alex/3dv_mockup/Venus/venus_part03.npy", ".", 0.1, [0.2], n_points)
	extractor2.extract()
	colors1 = ['blue' for _ in range(n_points)]
	colors2 = ['blue' for _ in range(n_points)]
	x_lines = []
	y_lines = []
	z_lines = []
	c = 1

	for i in range(n_points):
		for j in range(n_points):
			d = np.linalg.norm(extractor1.cov_mats[i] - extractor2.cov_mats[j], ord='fro')
			if d < 0.3:
				colors1[i] = 'green'
				colors2[j] = 'green'

				x_lines.append(extractor1.keypoints[i][0])
				x_lines.append(extractor2.keypoints[j][0]+c)
				x_lines.append(None)

				y_lines.append(extractor1.keypoints[i][1])
				y_lines.append(extractor2.keypoints[j][1])
				y_lines.append(None)

				z_lines.append(extractor1.keypoints[i][2])
				z_lines.append(extractor2.keypoints[j][2])
				z_lines.append(None)

	fig = go.Figure()
	fig.add_trace(
		go.Scatter3d(
			x = extractor1.point_cloud[:, 0],
			y = extractor1.point_cloud[:, 1],
			z = extractor1.point_cloud[:, 2],
			mode='markers',
			marker=dict(
				size=1,
			)
		)
	)
	fig.add_trace(
		go.Scatter3d(
			x = extractor1.keypoints[:, 0],
			y = extractor1.keypoints[:, 1],
			z = extractor1.keypoints[:, 2],
			mode='markers',
			marker=dict(
				size=3,
				color=colors1
			)
		)
	)
	extractor2.keypoints[:, 0] += c
	extractor2.point_cloud[:, 0] += c
	fig.add_trace(
		go.Scatter3d(
			x = extractor2.point_cloud[:, 0],
			y = extractor2.point_cloud[:, 1],
			z = extractor2.point_cloud[:, 2],
			mode='markers',
			marker=dict(
				size=1,
			)
		)
	)
	fig.add_trace(
		go.Scatter3d(
			x = extractor2.keypoints[:, 0],
			y = extractor2.keypoints[:, 1],
			z = extractor2.keypoints[:, 2],
			mode='markers',
			marker=dict(
				size=3,
				color=colors2
			)
		)
	)

	fig.add_trace(
		go.Scatter3d(
			x = x_lines,
			y = y_lines,
			z = z_lines,
			mode='lines',
		)
	)
	fig.show()
	

	# fragments = os.listdir(args.dataset_dir)
	# fragments = [x for x in fragments if x.endswith(".npy")]

	# f_args = []

	# for fragment in fragments:
	# 	print(f"Fragment: {fragment}")
	# 	fragment_path = os.path.join(args.dataset_dir, fragment)
	# 	keypoints_dir = os.path.join(args.dataset_dir, "keypoints")
	# 	if not os.path.exists(keypoints_dir):
	# 		os.mkdir(keypoints_dir)

	# 	output_path = os.path.join(keypoints_dir, fragment)
	# 	f_args.append((fragment_path, output_path, args.keypoint_radius, args.r_vals, args.n_keypoints))
	# with Pool(processes=8) as pool:
	# 	pool.starmap(f, f_args)