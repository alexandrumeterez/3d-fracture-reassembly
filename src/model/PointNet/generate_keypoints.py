import numpy as np
import os
import open3d as o3d
import shutil
from joblib import Parallel, delayed

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# DATA_PATH = 
DATA_PATH = "/home/sombit/objects_with_des/5_shapes_4-1_seeds/"

def handle_folder(folder):
    # initiate the paths and a log
    log = []
    fragment_path = os.path.join(DATA_PATH, folder)
    kpts_path = os.path.join(DATA_PATH, folder, 'keypoints_cls')

    # clean keypoints
    if os.path.exists(kpts_path):
        shutil.rmtree(kpts_path)
    os.makedirs(kpts_path)

    # generate keypoints for each fragment
    for file in os.listdir(fragment_path):
        if file.endswith('.npy'):
            # load the x y u coordinates
            fragment = np.load(os.path.join(fragment_path, file))
            point_cloud = fragment[:, :3]
            
            # generate a pointcloud and find iss keypoints
            # this assumes that there are enough vertices in the cloud
            # to get meaningful information            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            output = o3d.geometry.keypoint.compute_iss_keypoints(pcd, min_neighbors=4)
            output = [i for i in output.points]
            print(len(output))
            # save the keypoints
            filename = file.split('cleaned')[0] + "kpts_"+file.split('.')[-2] + ".npy"
            np.save(os.path.join(kpts_path, filename), output)

            # write to the log
            folder_path = os.path.join(DATA_PATH, folder)
            log_path = os.path.join(folder_path, 'log.txt')
            log.append(f'{filename} : {len(output)}\n')

            with open(log_path, "a") as text_file:
                text_file.write(''.join(log))
            
    print("Done with folder: ", folder)


def main():  
    folders = os.listdir(DATA_PATH)
    Parallel(n_jobs=8)(delayed(handle_folder)(folder) for folder in folders)

if __name__ == '__main__':
    main()
