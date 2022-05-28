fragments=($(ls $1 | sort))
n=${#fragments[@]}
for (( i=0; i<$n; ++i ))
do
    for (( j=i+1; j<$n; ++j ))
    do
        fragment1="$1/${fragments[i]}"
        fragment2="$1/${fragments[j]}"
        command="python3 classical_method.py --fragment1 $fragment1 --fragment2 $fragment2 --n_keypoints 512 --keypoint_radius 0.2 --r_vals 0.04 0.05 0.06 0.08 0.10 --threshold 0.2 --mode 2 --nms --nms_rad 0.04"
        echo $command
        eval $command
    done
done
