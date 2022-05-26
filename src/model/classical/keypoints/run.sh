folders=$(ls /cluster/project/infk/courses/252-0579-00L/group_reassembly/data/ | sort)

for folder in $folders
do
	dataset_dir="/cluster/project/infk/courses/252-0579-00L/group_reassembly/data/$folder"
	echo $dataset_dir

	command="python3 classical_method.py --dataset_dir $dataset_dir --n_keypoints 512 --keypoint_radius 0.04 --r_vals 0.04 0.05 0.06 0.08 0.10 --threshold 0.2 --mode 1 --nms --nms_rad 0.04"

	job_sub="bsub -R rusage[mem=6000] $command"

	eval $job_sub
done
