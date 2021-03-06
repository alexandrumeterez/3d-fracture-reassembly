# cluster command: bsub -R "rusage[mem=8000]" ./blender --background --python blender_auto_fracture_import.py
"""
USAGE:
    For use on the ETH Euler Cluster's batch system:
    Call this file from commandline to generate a job for each object located in the 'script-input' folder inside the blender directory
    arguments: --shard_count <int> --seed_count <int>

    needs the file 'blender_fracture_single.py' in the same directory as the blender executable
"""
import os, subprocess, argparse

input_dir = os.path.abspath("script-input")


def spawn_jobs(shard_count, seed_count):
    files = os.listdir(input_dir)
    for i, file in enumerate(files):
        print(f"Spawning Process {i} of {len(files)}")
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
        description="Spawn jobs for blender_fracture_single.py"
    )
    parser.add_argument(
        "--shard_count", type=int, help="Number of shards to split the model into", required=True
    )
    parser.add_argument(
        "--seed_count", type=int, help="Number of seeds to use for each shard", required=True
    )
    args = parser.parse_args()
    spawn_jobs(args.shard_count, args.seed_count)
