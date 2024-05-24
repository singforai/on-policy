#! /bin/bash


# # Original
# command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name Step_cost_Mappo --group_name Step_cost_Mappo --use_reward_shaping --use_shaping_weight"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
# parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

# wait 

#clusterized_reward_shaping
command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name clusterized_shaping_MAPPO_v2 --group_name clusterized_shaping_MAPPO_v2 --num_clusters 20" 

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

wait 



