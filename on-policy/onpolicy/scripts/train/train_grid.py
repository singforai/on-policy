#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import gym
import torch
import onpolicy

from gym import spaces
from onpolicy.config import get_config
from onpolicy.envs.ma_gym.magym_env import MagymEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.utils.util import MultiAgentObservationSpace
from gym.spaces import Box

"""Train script for MPEs."""

def make_obs_id(env):
    num_agents = env.n_agents
    observation_space = MultiAgentObservationSpace(
        [spaces.Box(
            np.concatenate((env._obs_low, np.array([0]))), 
            np.concatenate((env._obs_high, np.array([1]))))
        for _ in range(num_agents)]
    )
    _obs_high = np.tile(np.concatenate((env._obs_low, np.array([0]))), num_agents)
    _obs_low = np.tile(np.concatenate((env._obs_high, np.array([1]))), num_agents)
    share_observation_space = MultiAgentObservationSpace(
        [
            spaces.Box(_obs_high, _obs_low)
            for _ in range(num_agents)
        ]
    )   
    return observation_space, share_observation_space


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Checkers-v0":
                env = gym.make(
                    id="ma_gym:" + all_args.env_name,
                    full_observable=False,
                    max_steps=100,
                    step_cost=-0.01,
                )
                observation_space, share_observation_space = make_obs_id(env = env)
                env.observation_space = observation_space.copy()
                env.share_observation_space = share_observation_space.copy()
            
            elif all_args.env_name == "Switch2-v0":
                env = gym.make(
                    id = "ma_gym:" + all_args.env_name,
                    full_observable=False,
                    max_steps=50,
                    step_cost=-0.1,
                )
                observation_space, share_observation_space = make_obs_id(env = env)
                env.observation_space = observation_space.copy()
                env.share_observation_space = share_observation_space.copy()
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Checkers-v0":
                env = gym.make(
                    id="ma_gym:" + all_args.env_name,
                    full_observable=False,
                    max_steps=100,
                    step_cost=-0.01,
                )
                env.share_observation_space = env.observation_space
            elif all_args.env_name == "Switch2-v0":
                env = gym.make(
                    id = "ma_gym:" + all_args.env_name,
                    full_observable=False,
                    max_steps=50,
                    step_cost=-0.1,
                )
                env.share_observation_space = env.observation_space
         
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    # parser.add_argument("--user_name", type=str, default="sangkiko", help="username of wandb")

    all_args = parser.parse_known_args(args)[0]

    # all_args.use_wandb = True
    # all_args.env_name = "Checkers-v0"
    all_args.user_name = "singfor7012"
    all_args.n_rollout_threads = 1
    all_args.use_centralized_V = True
    print(all_args)

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.num_gpu}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            group= all_args.group_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.magym_runner import MagymRunner as Runner
    else:
        from onpolicy.runner.separated.magym_runner import MagymRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
