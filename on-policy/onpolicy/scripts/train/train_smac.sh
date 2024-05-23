#! /bin/bash


# # Original
# command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name Step_cost_Mappo --group_name Step_cost_Mappo --use_reward_shaping --use_shaping_weight"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
# parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

# wait 

# # Nonweighted_reward_shaping
# command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name Nonweighted_shaping_MAPPO --group_name Nonweighted_shaping_MAPPO --use_shaping_weight"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
# parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

# wait 

#sigmoid_flipsigmoid_Weighted_reward_shaping
command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name sigmoid_flipsigmoid_weighted_l1loss_shaping_MAPPO --group_name sigmoid_flipsigmoid_weighted_l1loss_shaping_MAPPO --shaping_weight_type sigmoid_flipsigmoid"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

wait 


#Sigmoid_weighted_reward_shaping
# command="python train_smac.py --seed {0} --env_name StarCraft2 --experiment_name Sigmoid_weighted_shaping_MAPPO --group_name Sigmoid_weighted_shaping_MAPPO --shaping_weight_type sigmoid"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 250 923 582 305 482 &
# parallel -j 10 $command $args_gpu1 ::: 234 92 350 293 753 

# wait 


