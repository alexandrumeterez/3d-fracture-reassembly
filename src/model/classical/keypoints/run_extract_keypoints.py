from extract_keypoints import SDExtractor
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_dir', type=str, required=True)
	parser.add_argument('--npoints_thresh', default=10000, type=int)
	parser.add_argument('--nkeypoints', default=512, type=int)
	parser.add_argument('--r1', default=0.1, type=float)
	parser.add_argument('--r2', default=0.09, type=float)

	args = parser.parse_args()

	objects = os.listdir(args.dataset_dir)
	for object in objects:
		path = os.path.join(args.dataset_dir, object)
		print(f"Running: {path}")

		extractor = SDExtractor(path)
		extractor.extract_keypoints(args.npoints_thresh, args.nkeypoints, args.r1, args.r2)