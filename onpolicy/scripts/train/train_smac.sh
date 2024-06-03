#! /bin/bash


# # Original
command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name Mappo --group_name Original --use_reward_shaping --map_name 2s3z --num_env_steps 4000000"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

wait 

#clusterized_reward_shaping
# command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name MAPPO --group_name EWMA_0999 --map_name 2s3z --num_env_steps 4000000 \
# --num_clusters 20 --num_pre_sampling 10000 --cluster_update_interval 100"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
# parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

# wait 



