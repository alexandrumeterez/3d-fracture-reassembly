import os
import argparse
import subprocess

def generate_command(root, detector_model_path, dataset_type='customnet'):
    command = f'bsub -R rusage[ngpus_excl_p=1,mem=30000] python save_keypoints.py --dataset_type {dataset_type} --root {root} --detector_model_path {detector_model_path}'
    return command

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type = str)
    parser.add_argument('--detector_model_path', type=str)
    args = parser.parse_args()

    main_folder = args.folder
    subdirs = os.listdir(main_folder)

    for d in subdirs:
        print(f"[INFO] Launching on: {d}")
        dir_path = os.path.join(main_folder, d)
        command = generate_command(root=dir_path, detector_model_path=args.detector_model_path)

        subprocess.run(command.split())