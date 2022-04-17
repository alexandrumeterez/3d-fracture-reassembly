import numpy as np
from scipy import spatial
import os
import argparse
import time
import plotly
import plotly.graph_objects as go
import open3d as o3d
class SDExtractor(object):
	def __init__(self, path, save, file):
		self.path = path
		self.save = save

		if not file:
			# Get all npy files
			self.files = os.listdir(self.path)
			self.files = [x for x in self.files if x.endswith('.npy')]

			self.n_shards = len(self.files)

			# Output dir
			self.output_path = os.path.join(self.path, 'keypoints')
			if not os.path.exists(self.output_path):
				os.mkdir(self.output_path)

	def compute_SD_point(self, neighbourhood, points, normals, p_idx):
		p_i = points[p_idx]
		n_p_i = normals[p_idx]
		p_i_bar = np.mean(points[neighbourhood], axis=0)
		v = p_i - p_i_bar
		SD = np.dot(v, -n_p_i)
		return SD

	# Assembling the above
	def get_SD_for_point_cloud(self, point_cloud, normals, r):
		n_points = len(point_cloud)
		tree = spatial.KDTree(point_cloud)
		# Compute SD
		SD = np.zeros((n_points))
		neighbourhoods = tree.query_ball_point(point_cloud, r, workers=-1)

		for i in range(n_points):
			neighbourhood = np.asarray(neighbourhoods[i])
			SD[i] = self.compute_SD_point(neighbourhood, point_cloud, normals, i)
		return SD

	def extract_keypoints(self, nkeypoints, r):
		for shard in range(0, self.n_shards):
			start = time.time()
			fragment_path = os.path.join(self.path, self.files[shard])
			print(f"Starting shard {shard}: {fragment_path}")

			fragment = np.load(fragment_path)
			point_cloud = fragment[:, :3]
			normals = fragment[:, 3:]
			
			SD = self.get_SD_for_point_cloud(point_cloud, normals, r=r)
			indices_to_keep = np.argsort(np.abs(SD))[-nkeypoints:]
			
			keypoints = point_cloud[indices_to_keep]
			kp_normals = normals[indices_to_keep]
			
			output = np.concatenate((keypoints, kp_normals), axis=1)
			output_fragment_path = os.path.join(self.output_path, self.files[shard])
			
			if self.save:
				np.save(output_fragment_path, output)
				print(f"Saved to: {output_fragment_path}")
			end = time.time()
			
			print(f"Time: {end-start}s")

	def extract_keypoints_shard(self, nkeypoints, r):
		fragment = np.load(self.path)
		point_cloud = fragment[:, :3]
		normals = fragment[:, 3:]
		
		SD = self.get_SD_for_point_cloud(point_cloud, normals, r=r)
		indices_to_keep = np.argsort(np.abs(SD))[-nkeypoints:]
		
		keypoints = point_cloud[indices_to_keep]
		kp_normals = normals[indices_to_keep]
				
		fig = go.Figure()
		fig.add_trace(
			go.Scatter3d(
				x = point_cloud[:, 0],
				y = point_cloud[:, 1],
				z = point_cloud[:, 2],
				mode='markers',
				marker=dict(
        			size=1,
    			)
			)
		)
		fig.add_trace(
			go.Scatter3d(
				x = keypoints[:, 0],
				y = keypoints[:, 1],
				z = keypoints[:, 2],
				mode='markers',
				marker=dict(
        			size=3,
    			)
			)
		)
		fig.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--object_dir', type=str, required=True)
	parser.add_argument('--nkeypoints', default=512, type=int)
	parser.add_argument('--r', default=0.1, type=float)
	parser.add_argument('--save', action='store_true')
	parser.add_argument('--file', action='store_true')

	args = parser.parse_args()

	extractor = SDExtractor(args.object_dir, args.save, args.file)

	if not args.file:
		extractor.extract_keypoints(args.nkeypoints, args.r)
	else:
		extractor.extract_keypoints_shard(args.nkeypoints, args.r)
		