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

				# For each point in the keypoint neighbourhood
				for idx, p_i in enumerate(keypoint_neighbourhood):
					if p_i == keypoint_index:
						continue
					phi_i = []

					# Get neighbourhood of p_i
					p_i_neighbourhood = neighbourhoods[r][p_i]

					# Compute cosines
					p_i_point = point_cloud[p_i]
					p_i_normal = normals[p_i]
					p_p_i_vector = p_i_point - keypoint
					cos_alpha = np.dot(p_p_i_vector, p_i_normal) / np.linalg.norm(p_p_i_vector)
					cos_beta = np.dot(p_p_i_vector, keypoint_normal) / np.linalg.norm(p_p_i_vector)
					cos_gamma = np.dot(p_i_normal, keypoint_normal)
					phi_i.append(cos_alpha)
					phi_i.append(cos_beta)
					phi_i.append(cos_gamma)

					# Compute C_p_i and delta1, delta2, delta3
					C_p_i = np.cov(point_cloud[p_i_neighbourhood].T) * len(p_i_neighbourhood)
					U, S, Vt = np.linalg.svd(C_p_i)
					lambda1, lambda2, lambda3 = S
					delta1 = (lambda1 - lambda2) / lambda1
					delta2 = (lambda2 - lambda3) / lambda1
					delta3 = lambda3 / lambda1
					phi_i.append(delta1)
					phi_i.append(delta2)
					phi_i.append(delta3)

					# Compute H
					p_i_neighbourhood_points = point_cloud[p_i_neighbourhood]
					xdata = p_i_neighbourhood_points[:, :2]
					ydata = p_i_neighbourhood_points[:, 2]
					popt, _ = curve_fit(objective, xdata, ydata, p0=[0,0,0,0,0,0])
					a0, a1, a2, a3, a4, a5 = popt
					r_x = np.zeros((3))
					r_xy = np.zeros((3))
					r_xx = np.zeros((3))
					r_y = np.zeros((3))
					r_yy = np.zeros((3))
					r_x[2] = 2 * a0 * p_i_point[0] + a2 * p_i_point[1] + a3
					r_xx[2] = 2 * a0
					r_xy[2] = a2
					r_y[2] = 2 * a1 * p_i_point[1] + a2 * p_i_point[0] + a4
					r_yy[2] = 2 * a1
					r_x[0] = 1
					r_y[1] = 1
					E = np.dot(r_x, r_x)
					F = np.dot(r_x, r_y)
					G = np.dot(r_y, r_y)
					L = np.dot(r_xx, p_i_normal)
					M = np.dot(r_xy, p_i_normal)
					N = np.dot(r_yy, p_i_normal)
					H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F ** 2))

					# Set the value
					phi_i.append(H)

					Phi.append(phi_i)
				Phi = np.array(Phi)
				C_r = np.cov(Phi.T)

				# Compute log and save
				S, R = np.linalg.eig(C_r)
				S += 1e-3
				S = np.log(S)
				log_C_r = R @ np.diag(S) @ R.T
				output_matrix[n_keypoint, 6 + r_idx * 49: 6 + (r_idx + 1) * 49] = log_C_r.ravel()

		np.save(self.output_path, output_matrix)
def f(fragment_path, output_path, keypoint_radius, r_vals, n_keypoints):
	extractor = Extractor(fragment_path, output_path, keypoint_radius, r_vals, n_keypoints)
	extractor.extract()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_dir", type=str)
	parser.add_argument("--keypoint_radius", type=float, required=True)
	parser.add_argument("--n_keypoints", type=int, required=True)
	parser.add_argument("--r_vals", nargs='+', required=True, type=float)

	args = parser.parse_args()
	fragments = os.listdir(args.dataset_dir)
	fragments = [x for x in fragments if x.endswith(".npy")]

	f_args = []

	for fragment in fragments:
		print(f"Fragment: {fragment}")
		fragment_path = os.path.join(args.dataset_dir, fragment)
		keypoints_dir = os.path.join(args.dataset_dir, "keypoints")
		if not os.path.exists(keypoints_dir):
			os.mkdir(keypoints_dir)

		output_path = os.path.join(keypoints_dir, fragment)
		f_args.append((fragment_path, output_path, args.keypoint_radius, args.r_vals, args.n_keypoints))
	with Pool(processes=8) as pool:
		pool.starmap(f, f_args)