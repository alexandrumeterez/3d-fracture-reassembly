import numpy as np
from scipy import spatial
import os
import argparse
import time
import scipy
from scipy.optimize import curve_fit
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

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

		# Get all radius r neighbourhoods for each r
		keypoint_radius = self.keypoint_radius
		tree = spatial.KDTree(point_cloud)

		# Extract keypoints
		neighbourhood = tree.query_ball_point(point_cloud, keypoint_radius, workers=-1)
		SD = self.get_SD_for_point_cloud(point_cloud, normals, neighbourhood)

		# Extract keypoint indices
		keypoint_indices = np.argsort(np.abs(SD))[-self.n_keypoints:]

		# Extract keypoints
		keypoints = point_cloud[keypoint_indices]
		keypoints_normals = normals[keypoint_indices]
		# For each r value
		s = len(self.r_vals)

		# Output matrix
		output_matrix = np.zeros((len(keypoints), 3 + 3 + s * 49))
		output_matrix[:, :3] = keypoints
		output_matrix[:, 3:6] = keypoints_normals

		for r_idx, r in enumerate(self.r_vals):
			print(f"Fragment {fragment_path} with radius {r}")
			r_neighbourhood = tree.query_ball_point(point_cloud, r, workers=-1)

			# For each point p_i in the nbhd of a keypoint p, compute the matrix and the features
			for p_idx in range(len(keypoints)):
				# The matrix with all the features, where each row is a phi_i (point in nbhd of p)
				Phi = np.zeros((len(r_neighbourhood[keypoint_indices[p_idx]]), 7))
				
				p = keypoints[p_idx]
				p_neighbourhood = np.asarray(r_neighbourhood[keypoint_indices[p_idx]])
				# Compute C_p
				p_bar = np.mean(point_cloud[p_neighbourhood], axis=1)
				P = point_cloud[p_neighbourhood].T
				C_p = (P - p_bar) @ (P - p_bar).T

				# Do SVD on C_p
				U, S, Vt = np.linalg.svd(C_p)
				lambda1, lambda2, lambda3 = S

				# Compute deltas and save them in matrix
				delta1 = (lambda1 - lambda2) / lambda1
				delta2 = (lambda2 - lambda3) / lambda1
				delta3 = lambda3 / lambda1
				Phi[:, 3] = delta1
				Phi[:, 4] = delta2
				Phi[:, 5] = delta3
				# Compute cosine for each keypoint (vectorized, p_i are the points in the nbhd of p)
				# print(point_cloud[p_neighbourhood].shape, p.shape)
				p_pi = point_cloud[p_neighbourhood] - p
				n_pi = normals[p_neighbourhood]
				n_p = normals[keypoint_indices[p_idx]]
				for i in range(len(p_neighbourhood)):					
					cos_alpha_i = np.dot(p_pi[i], n_pi[i]) / np.linalg.norm(p_pi)
					cos_beta_i = np.dot(p_pi[i], n_p) / np.linalg.norm(p_pi)
					cos_gamma_i = np.dot(n_pi[i], n_p)

					Phi[i, 0] = cos_alpha_i
					Phi[i, 1] = cos_beta_i
					Phi[i, 2] = cos_gamma_i
				# Compute H
				xdata = point_cloud[p_neighbourhood][:, :2]
				ydata = point_cloud[p_neighbourhood][:, 2]


				# define the true objective function
				def objective(X, a0, a1, a2, a3, a4, a5):
					x = X[:, 0]
					y = X[:, 1]
					return a0 * (x ** 2) + a1 * (y ** 2) + a2 * (x * y) + a3 * x + a4 * y + a5

				popt, _ = curve_fit(objective, xdata, ydata, p0=[0,0,0,0,0,0])
				a0, a1, a2, a3, a4, a5 = popt

				# (x, y, f(x, y))
				# r_x = (1, 0, d/dx f)
				# r_xy = (0, 0, d/dxdy f)
				# r_y = (0, 1, d/dy f)
				# r_xx = (0, 0, d/dxdx f)
				# r_yy = (0, 0, d/dydy f)
				
				r_x = np.zeros((3))
				r_xy = np.zeros((3))
				r_xx = np.zeros((3))
				r_y = np.zeros((3))
				r_yy = np.zeros((3))
				
				r_x[2] = 2 * a0 * p[0] + a2 * p[1] + a3
				r_xx[2] = 2 * a0
				r_xy[2] = a2
				r_y[2] = 2 * a1 * p[1] + a2 * p[0] + a4
				r_yy[2] = 2 * a1
				r_x[0] = 1
				r_y[1] = 1
				E = np.dot(r_x, r_x)
				F = np.dot(r_x, r_y)
				G = np.dot(r_y, r_y)
				L = np.dot(r_xx, n_p)
				M = np.dot(r_xy, n_p)
				N = np.dot(r_yy, n_p)
				H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F ** 2))

				# Set the value
				Phi[:, 6] = H

				# Now we have Phi, gotta compute the last part (cov mat)
				Phi_bar = np.mean(Phi, axis=0)
				centered_Phi = Phi - Phi_bar
				C_r = centered_Phi.T @ centered_Phi / (len(p_neighbourhood))
				output_matrix[p_idx, 6 + r_idx * 49: 6 + (r_idx + 1) * 49] = C_r.ravel()
		# print(output_matrix)
		output_matrix[:, 6:] = np.log(output_matrix[:, 6:])
		
		np.save(self.output_path, output_matrix)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_dir", type=str)
	parser.add_argument("--keypoint_radius", type=float, required=True)
	parser.add_argument("--n_keypoints", type=int, required=True)
	parser.add_argument("--r_vals", nargs='+', required=True, type=float)

	args = parser.parse_args()
	fragments = os.listdir(args.dataset_dir)
	fragments = [x for x in fragments if x.endswith(".npy")]

	for fragment in fragments:
		print(f"Fragment: {fragment}")
		fragment_path = os.path.join(args.dataset_dir, fragment)
		keypoints_dir = os.path.join(args.dataset_dir, "keypoints")
		if not os.path.exists(keypoints_dir):
			os.mkdir(keypoints_dir)

		output_path = os.path.join(keypoints_dir, fragment)
		extractor = Extractor(fragment_path, output_path, args.keypoint_radius, args.r_vals, args.n_keypoints)
		extractor.extract()