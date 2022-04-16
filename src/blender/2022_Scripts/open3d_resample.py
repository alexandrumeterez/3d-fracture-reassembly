import numpy as np
import open3d as o3d
import os
import plotly.graph_objects as go

from create_neighbor_LUT import generate_neighbors

"""
Script to convert Pointclouds with high density
Downsamples pointcloud, saves a html plot and generates a csv with neighboring information

Folder structure:
    '-root
        '-npy
            '-OUTPUT FOLDERS
        '-ply
            '-INPUT FOLDERS

INPUT FORMAT: .ply pointcloud with normals
OUTPUT FORMAT: .npy array with [x y z nx ny nz] values

Loops over all subfolders of root/input_folder and outputs to same folder in root/output_folder
Fragments with <1000 points after downsampling won't get downsampled
"""

root = "PC_Artificial/default_cube"
input_folder = f"{root}/ply"  # should contain pointclouds in .ply format
output_folder = f"{root}/npy" # will be generated for output
sample_size = 0.05


def convert_pointclouds(input_root, output_root):
    """
    Entry point to loop over subfolders
    """
    folders = [
        x for x in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, x))
    ]
    print(f"Folders: {folders}")
    for dir in folders:
        collect_pointclouds(dir, input_root, output_root)


def collect_pointclouds(dir, input_root, output_root):
    """
    convert and save all pointcloud data from a single folder
    """
    files = [
        x[:-4] for x in os.listdir(os.path.join(input_root, dir)) if x.endswith(".ply")
    ]
    pc_dict = {}
    b_max = np.zeros((3,))
    b_min = np.zeros((3,))

    for file in files:
        pc = o3d.io.read_point_cloud(os.path.join(input_root, dir, f"{file}.ply"))
        b_max = np.max([pc.get_max_bound(), b_max], axis=0)
        b_min = np.min([pc.get_min_bound(), b_min], axis=0)
        print(
            f"Reading {file}, Max Bound:{b_max}, Min Bound: {b_min}" + 30 * " ",
            end="\r",
        )
        pc_dict[file] = pc
    print("\n")
    scale = np.max(b_max - b_min)
    if scale < 0.01:
        assert f"SCALE TOO SMALL -- Folder: {dir}"
    print(f"Maximal Scale: {scale}")
    _downsample_and_save(dir, pc_dict, output_root, scale)

    # Save neighbor information to csv, super slow for large point clouds
    # neighbors = generate_neighbors(os.path.join(output_root, dir))
    # head_str = f"Neighboring Matrix of Fragments 0-{len(files)} from {dir}\n {files}"
    # np.savetxt(os.path.join(output_root, dir, "neighbors.csv"), neighbors, header=head_str, fmt='%i')


def _downsample_and_save(folder, pc_dict, output_root, scale=2):
    """
    helper for downsamplng and plotting
    """
    # Resample and save to .npy, also output a html scatterplot for easy visualization
    if not os.path.exists(os.path.join(output_root, folder)):
        os.makedirs(os.path.join(output_root, folder))

    fig = go.Figure()
    for file, pcloud in pc_dict.items():
        pcloud.scale(2 / scale, [0, 0, 0])
        pc_d = pcloud.voxel_down_sample(voxel_size=sample_size)

        points = np.asarray(pc_d.points)
        normals = np.asarray(pc_d.normals)
        if points.shape[0] < 1000:
            print(f"{file} resampled to < 1000 points, using original")
            points = np.asarray(pcloud.points)
            normals = np.asarray(pcloud.normals)
        assert (
            points.shape[0] >= 1000
        ), f"Shard {file} has less than 1000 points: {points.shape[0]}"

        data = np.concatenate([points, normals], axis=1)

        fig.add_trace(
            go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                name=f"{file}.npy",
                mode="markers",
                marker=dict(size=2),
            )
        )

        out_file = os.path.join(output_root, folder, file)
        # print(f"Saving {out_file}: {pc_d}")
        np.save(out_file, data)
    fig.write_html(os.path.join(output_root, folder, "scatter_plot.html"))


if __name__ == "__main__":
    convert_pointclouds(input_folder, output_folder)
