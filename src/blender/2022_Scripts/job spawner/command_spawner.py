# example cluster command: bsub -R "rusage[mem=8000]" ./blender --background --python blender_auto_fracture_import.py
"""
USAGE:
    call this file from commandline to generate a job for each object located in the 'script-input' folder inside rthe blender directory
    arguments: --shard_count <int> --seed_count <int>

    needs the file 'blender_fracture_single.py' in the same folder inside the blnder directory
"""

import os, subprocess, argparse

input_dir = os.path.abspath("script-input")


def spawn_jobs(shard_count, seed_count):
    i = 0
    files = os.listdir(input_dir)
    for file in files[:2]:
        print(f"Spawning Process {i} of {len(files)}")
        i+=1
        subprocess.run(
            [
                "bsub",
                "-R",
                "rusage[mem=8000]",
                "./blender",
                "--background",
                "--python",
                "blender_fracture_single.py",
                "--",
                str(shard_count),
                str(seed_count),
                file,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spawn jobs for blender_auto_fracture_cluster.py"
    )
    parser.add_argument(
        "--shard_count", type=int, help="Number of shards to split the model into", required=True
    )
    parser.add_argument(
        "--seed_count", type=int, help="Number of seeds to use for each shard", required=True
    )
    args = parser.parse_args()
    spawn_jobs(args.shard_count, args.seed_count)
