
folders=$(ls /cluster/project/infk/courses/252-0579-00L/group_reassembly/data/ | sort)

for folder in $folders
do
	dataset_dir="/cluster/project/infk/courses/252-0579-00L/group_reassembly/data/$folder"
	echo $dataset_dir
  voxel_size=0.03
  command="python subsample.py --dataset-dir $dataset_dir --voxel-size $voxel_size"
	job_sub="bsub -R rusage[mem=6000] $command"

	eval $job_sub
done
