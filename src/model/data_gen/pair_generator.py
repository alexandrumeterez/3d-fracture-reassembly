import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import time
from pathlib import Path
import argparse

class Generator(object):
	def __init__(self, input_dir, output_dir):
		self.input_dir = input_dir
		self.output_dir = output_dir

		Path(f"{self.output_dir}/connected_1vN/fragments_1").mkdir(parents=True, exist_ok=True)
		Path(f"{self.output_dir}/connected_1vN/fragments_2").mkdir(parents=True, exist_ok=True)
		Path(f"{self.output_dir}/flipped_normals/fragments_1").mkdir(parents=True, exist_ok=True)
		Path(f"{self.output_dir}/flipped_normals/fragments_2").mkdir(parents=True, exist_ok=True)
	
	def generate(self, max_parts=1000, threshold=1e-3, needed_matches=100, sigma=0.001):
		names1_1vN=[]
		names2_1vN=[]
		names_FN=[]
		fragment = []
		n_loaded_parts = -1

		m = 0
		p = 0

		for part in range(max_parts):
			try:
				fragment.append(np.load(f'{self.input_dir}/{part}.npy'))
			except:
				n_loaded_parts = part
				print(f'[INFO] {n_loaded_parts} parts loaded')
				break
		
		print('[INFO] Saving connected 1vN')
		for i in range(n_loaded_parts - 1):
			fragment_set=np.zeros((1,6))
			names2_temp=[]
			for j in range(i+1, n_loaded_parts):
				# search for corresponding points in two parts (distance below a treshold)
				matches = np.count_nonzero(cdist(fragment[i][:,:3],fragment[j][:,:3]) < threshold)
				print(f'[INFO] Fragments ({i} - {j}) have {matches} matches')

				# if there are more than 100 matches, the parts are considered neighbours
				if matches > needed_matches:
					names2_temp.append(f'{self.input_dir}/{j}.npy')
					# all matching parts are concatenated together in one file, as if they where one point cloud
					fragment_set=np.concatenate((fragment_set, fragment[j]),axis=0)
				
			# delete row of zeros at the beginning
			fragment_set=np.delete(fragment_set, 0, axis=0)

			# only generate a trainingpair, if neighbouring part was found
			if fragment_set.shape[0] != 0:
				# generate list of used parts
				names1_1vN.append(f'{self.input_dir}/{i}.npy')
				names2_1vN.append(names2_temp)

				# add random displacement of points with a gaussian profile
				noise_i = np.random.default_rng().normal(0,sigma,fragment[i].shape)
				noise_set = np.random.default_rng().normal(0,sigma,fragment_set.shape)

				# save the two parts of the pairs as seperate .npy files
				np.save(f'{self.output_dir}/connected_1vN/fragments_1/{m}.npy', fragment[i]+noise_i)
				np.save(f'{self.output_dir}/connected_1vN/fragments_2/{m}.npy', fragment_set+noise_set)
				m += 1

		# save pairs with flipped surface normals
		print('[INFO] Saving flipped normals')
		for i in range(n_loaded_parts):
			# copy fragment, change the sign of the surface normals
			fragment_neg=np.copy(fragment[i])
			fragment_neg[:,3:6] = -fragment_neg[:,3:6]

			# generate list of used parts
			names_FN.append(f'{self.output_dir}/{i}.npy')

			# add random displacement of points with a gaussian profile
			noise_i=np.random.default_rng().normal(0,sigma,fragment[i].shape)
			noise_neg=np.random.default_rng().normal(0,sigma,fragment_neg.shape)

			# save the two parts of the pairs as seperate .npy files
			np.save(f'{self.output_dir}/flipped_normals/fragments_1/{p}.npy', fragment[i]+noise_i)
			np.save(f'{self.output_dir}/flipped_normals/fragments_2/{p}.npy', fragment_neg+noise_neg)
			p += 1
		
		pd.DataFrame(names1_1vN).to_csv(f'{self.output_dir}/connected_1vN/fragments_1.csv', index=False, header=False)
		pd.DataFrame(names2_1vN).to_csv(f'{self.output_dir}/connected_1vN/fragments_2.csv', index=False, header=False)
		pd.DataFrame(names_FN).to_csv(f'{self.output_dir}/flipped_normals/fragments.csv', index=False, header=False)

		print("Done")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	args = parser.parse_args()

	generator = Generator(args.input_dir, args.output_dir)
	generator.generate()