#! /bin/bash


# # Original
# command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name Mappo --group_name Mappo_original --use_reward_shaping"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
# parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

# wait 

#clusterized_reward_shaping
command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name MAPPO --group_name interval100_50_cluster20 \
--num_clusters 20 --num_pre_sampling 100 --cluster_update_interval 50"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

wait 



