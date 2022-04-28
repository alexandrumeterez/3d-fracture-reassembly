from turtle import color
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

if __name__ == '__main__':
	shard = 2
	fragment_path = f'/Users/alex/3dv_mockup/Venus/venus_part0{shard}.npy'
	fragment = np.load(fragment_path)
	point_cloud = fragment[:, :3]
	normals = fragment[:, 3:].T
	tree = spatial.KDTree(point_cloud)

	# params
	k = 150

	lbd = 2

	dists, nbhd = tree.query(x=point_cloud, k=k, workers=-1)
	labels = []
	colors = []

	for i in range(point_cloud.shape[0]):
		# get neighboring points
		closest_points = nbhd[i]
		C_i = np.mean(point_cloud[closest_points], axis=0)
		Z_i = dists[i][1]
		
		score = np.linalg.norm(C_i - point_cloud[i])
		if(score > lbd * Z_i):
			colors.append("red")
		else:
			colors.append("blue")
		labels.append(score > lbd * Z_i)
	
	labels = np.asarray(labels)

	fig = go.Figure()
	fig.add_trace(
		go.Scatter3d(
			x = point_cloud[:, 0],
			y = point_cloud[:, 1],
			z = point_cloud[:, 2],
			mode='markers',
			marker=dict(
				size=3,
				color=colors
			)
		)
	)
	fig.show()