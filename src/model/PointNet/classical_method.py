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
import pandas as pd
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

def objective(X, a0, a1, a2, a3, a4, a5):
	x = X[:, 0]
	y = X[:, 1]
	return a0 * (x ** 2) + a1 * (y ** 2) + a2 * (x * y) + a3 * x + a4 * y + a5

class Extractor(object):
	def __init__(self, fragment_path, output_path, keypoint_radius, r_values, n_keypoints, nms, nms_rad):
		self.fragment_path = fragment_path
		self.output_path = output_path
		self.r_vals = r_values
		self.keypoint_radius = keypoint_radius
		self.n_keypoints = n_keypoints
		self.cov_mats = []
		self.point_cloud = None
		self.keypoints = None
		self.nms = nms
		self.nms_rad = nms_rad

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
		self.point_cloud = point_cloud
		normals = fragment[:, 3:]
		point_cloud = np.asarray(point_cloud, order='F')
		normals = np.asarray(normals, order='F')

		# Get all radius r neighbourhoods for each r
		keypoint_radius = self.keypoint_radius
		tree = spatial.KDTree(point_cloud)

		# Extract keypoints
		nbhd = tree.query_ball_point(point_cloud, keypoint_radius, workers=-1)
		SD = self.get_SD_for_point_cloud(point_cloud, normals, nbhd)

		# Fix normals
		normals = normals * np.sign(SD[:, None])

		# Extract keypoint indices
		# Perfrom Non-maximal Suppression
		if self.nms:
			keypoint_indices = np.argsort(-np.abs(SD))[:self.n_keypoints*15]
			keypoint_indices_nms = []
			rem_list = []
			nms_rad = self.nms_rad
			nbhd_nms = tree.query_ball_point(point_cloud, nms_rad, workers=-1)

			for i in keypoint_indices:
				if i not in rem_list:
					rem_list.extend(nbhd_nms[i])
					keypoint_indices_nms.append(i)

			keypoint_indices = np.asarray(keypoint_indices_nms)[:self.n_keypoints]
		else:
			keypoint_indices = np.argsort(np.abs(SD))[-self.n_keypoints:]
		

		self.keypoints = self.point_cloud[keypoint_indices]

		# Compute the neighbourhoods in all r vals 
		neighbourhoods = {}
		for r in self.r_vals:
			neighbourhoods[r] = tree.query_ball_point(point_cloud, r, workers=-1)
		
		# Output
		n_features_used = 3 # change this if you uncomment any of the rest
		output_matrix = np.zeros((len(keypoint_indices), 3 + 3 + n_features_used * n_features_used * len(self.r_vals)))

		# For each keypoint
		for n_keypoint, keypoint_index in enumerate(tqdm(keypoint_indices)):
			# Get keypoint and normal of the keypoint
			keypoint = point_cloud[keypoint_index]
			keypoint_normal = normals[keypoint_index]

			# Set output matrix
			output_matrix[n_keypoint, :3] = keypoint
			output_matrix[n_keypoint, 3:6] = keypoint_normal
			keypoint_cov_mats = []

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

					# # Compute C_p_i and delta1, delta2, delta3
					# C_p_i = np.cov(point_cloud[p_i_neighbourhood].T) * len(p_i_neighbourhood)
					# U, S, Vt = np.linalg.svd(C_p_i)
					# lambda1, lambda2, lambda3 = S
					# delta1 = (lambda1 - lambda2) / lambda1
					# delta2 = (lambda2 - lambda3) / lambda1
					# delta3 = lambda3 / lambda1
					# phi_i.append(delta1)
					# phi_i.append(delta2)
					# phi_i.append(delta3)

					# # Compute H
					# p_i_neighbourhood_points = point_cloud[p_i_neighbourhood]
					# xdata = p_i_neighbourhood_points[:, :2]
					# ydata = p_i_neighbourhood_points[:, 2]
					# popt, _ = curve_fit(objective, xdata, ydata, p0=[0,0,0,0,0,0])
					# a0, a1, a2, a3, a4, a5 = popt
					# r_x = np.zeros((3))
					# r_xy = np.zeros((3))
					# r_xx = np.zeros((3))
					# r_y = np.zeros((3))
					# r_yy = np.zeros((3))
					# r_x[2] = 2 * a0 * p_i_point[0] + a2 * p_i_point[1] + a3
					# r_xx[2] = 2 * a0
					# r_xy[2] = a2
					# r_y[2] = 2 * a1 * p_i_point[1] + a2 * p_i_point[0] + a4
					# r_yy[2] = 2 * a1
					# r_x[0] = 1
					# r_y[1] = 1
					# E = np.dot(r_x, r_x)
					# F = np.dot(r_x, r_y)
					# G = np.dot(r_y, r_y)
					# L = np.dot(r_xx, p_i_normal)
					# M = np.dot(r_xy, p_i_normal)
					# N = np.dot(r_yy, p_i_normal)
					# H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F ** 2))

					# # Set the value
					# phi_i.append(H)

					Phi.append(phi_i)
				Phi = np.array(Phi)
				C_r = np.cov(Phi.T)

				# Compute log and save
				S, R = np.linalg.eig(C_r)
				S = np.log(S)
				log_C_r = R @ np.diag(S) @ R.T
				keypoint_cov_mats.append(log_C_r)
				output_matrix[n_keypoint, 6 + r_idx * n_features_used * n_features_used: 6 + (r_idx + 1) * n_features_used * n_features_used] = log_C_r.ravel()
			self.cov_mats.append(keypoint_cov_mats)
		np.save(self.output_path, output_matrix)

def f(fragment_path, output_path, keypoint_radius, r_vals, n_keypoints, nms, nms_rad):
	extractor = Extractor(fragment_path, output_path, keypoint_radius, r_vals, n_keypoints, nms, nms_rad)
	extractor.extract()

def visualize_matches(extractor1, extractor2, n_points, n_scales, threshold):
	x_lines = []
	y_lines = []
	z_lines = []
	c = 1
	colors1 = ['blue' for _ in range(n_points)]
	colors2 = ['blue' for _ in range(n_points)]
	min_d = 99
	gt_cnt = 0
	inliers_cnt = 0
	outliers_cnt = 0
	gt_thres = 0.01

	# Calculate the ground Truth Matches
	for i in range(n_points):
		for j in range(n_points):
			if (np.linalg.norm(extractor1.keypoints[i] - extractor2.keypoints[j]) < gt_thres):
				gt_cnt+=1

				# x_lines.append(extractor1.keypoints[i][0])
				# x_lines.append(extractor2.keypoints[j][0]+c)
				# x_lines.append(None)

				# y_lines.append(extractor1.keypoints[i][1])
				# y_lines.append(extractor2.keypoints[j][1])
				# y_lines.append(None)

				# z_lines.append(extractor1.keypoints[i][2])
				# z_lines.append(extractor2.keypoints[j][2])
				# z_lines.append(None)

	print(f"No. of groundtruth matches : {gt_cnt}")		
			

	for i in range(n_points):
		for j in range(n_points):
			d = 0.0
			for s in range(n_scales):
				d += np.linalg.norm(extractor1.cov_mats[i][s] - extractor2.cov_mats[j][s], ord='fro')
			d /= n_scales
			if d < min_d:
				min_d = d
				# gt_dist = np.linalg.norm(extractor1.keypoints[i] - extractor2.keypoints[j])
				# print(f"Min: {min_d}", f"GT_dist: {gt_dist}")

			if d < threshold:
				colors1[i] = 'green'
				colors2[j] = 'green'

				gt_dist = np.linalg.norm(extractor1.keypoints[i] - extractor2.keypoints[j])
				print(f"Dist: {d}", f"GT_dist: {gt_dist}")
				if gt_dist < gt_thres:
					inliers_cnt+=1
				else:
					outliers_cnt+=1	

				x_lines.append(extractor1.keypoints[i][0])
				x_lines.append(extractor2.keypoints[j][0]+c)
				x_lines.append(None)

				y_lines.append(extractor1.keypoints[i][1])
				y_lines.append(extractor2.keypoints[j][1])
				y_lines.append(None)

				z_lines.append(extractor1.keypoints[i][2])
				z_lines.append(extractor2.keypoints[j][2])
				z_lines.append(None)

	print(f"No. of inliers: {inliers_cnt}")		
	print(f"No. of outliers: {outliers_cnt}")		

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
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default=2, type=int)
	parser.add_argument("--dataset_dir", type=str)
	parser.add_argument("--keypoint_radius", type=float, required=True)
	parser.add_argument("--n_keypoints", type=int, required=True)
	parser.add_argument("--r_vals", nargs='+', required=True, type=float)
	parser.add_argument("--threshold", type=float, default=1.0, required=True)
	parser.add_argument("--fragment1", type=str)
	parser.add_argument("--fragment2", type=str)
	parser.add_argument("--nms", action='store_true')
	parser.add_argument("--nms_rad", type=float, default=0.05)

	args = parser.parse_args()

	if args.mode == 1:
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
			f_args.append((fragment_path, output_path, args.keypoint_radius, args.r_vals, args.n_keypoints, args.nms, args.nms_rad))
		with Pool(processes=2) as pool:
			pool.starmap(f, f_args)
	elif args.mode == 2:
		# Just for visualization purposes

		r = args.keypoint_radius
		scales = args.r_vals
		n_points = args.n_keypoints
		extractor1 = Extractor(args.fragment1, "temp", r, scales, n_points, args.nms, args.nms_rad)
		extractor1.extract()
		extractor2 = Extractor(args.fragment2, "temp", r, scales, n_points, args.nms, args.nms_rad)
		extractor2.extract()

		visualize_matches(extractor2, extractor1, n_points, len(scales), args.threshold)
