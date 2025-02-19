import argparse
import os
from IPython.display import display
import numpy as np
import gymnasium as gym
import pandas as pd
import torch
import edss
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

import tensorboard

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=str, default="./experiments/test1.yaml", help="path to design to synthesize")
    parser.add_argument("--bayes-transform", type=str, default="arctan", help="transform for build progress for Bayes wrapper")
    parser.add_argument("--seed", type=int, default=145, help="seed for RNG")
    parser.add_argument("--num-envs", type=int, default=4, help="number of parallel simulation environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="number of steps to run in each environment per policy rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="number of epoch when optimizing loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="general advangage estimation lambda")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="entropy coefficient")
    parser.add_argument("--training-time", type=int, default=int(1e8), help="number of timesteps for training")
    parser.add_argument("--retrain", type=bool, default=False, help="Retrain agents")
    # TODO: device cpu vs cuda vs mps vs other
    args = parser.parse_args()
    args.device = "cpu"
    args.exp_dir = args.design.removesuffix('.yaml')
    return args

def make_EDSSEnv(design, filter=None, transform=None):
    def thunk():
        env = edss.EDSSEnv(design)
        match filter:
            case None:
                pass
            case "pomdp":
                env = edss.POMDPWrap(env)
            case "bayes":
                env = edss.BayesWrap(edss.POMDPWrap(env), transform)                
        return env
    return thunk

class BayesEstimator():
    def __init__(self, env: gym.Env, transform=None) -> None:
        self.buildActions: list[edss.Action] = []
        self.testActions: list[edss.Action] = []
        for action in env.unwrapped.actions:
            match action.type:
                case "build":
                    self.buildActions.append(action)
                case "test":
                    self.testActions.append(action)
                case _:
                    pass
        self.transform = transform
        self.features = np.zeros_like(env.unwrapped.features, dtype=np.single)

    def estimate(self, observation):
        self.features = np.zeros_like(self.features)
        for i, b in enumerate(self.buildActions):
            feature = b.feature
            self.features[feature] = max(self.features[feature], observation["Build_progress"][i])
        
        match self.transform:
            case "arctan":
                self.features = 2*np.arctan(self.features)/np.pi 
            case "tanh":
                self.features = np.tanh(0.55 * self.features)   # Note: ln(3)/2 ~ 0.55
            case _: 
                self.features = np.clip(self.features,1,0.5)
        
        for i, t in enumerate(self.testActions):
            test = observation["Test_progress"][i]
            sen = t.attr2
            spc = t.attr1
            feature = t.feature
            prior = self.features[feature]
            if test == -1:
                pass
            elif test == 1:
                self.features[feature] = (sen * prior) / (sen * prior + (1-spc) * (1 - prior))
            else: # test == 0
                self.features[feature] = ((1-sen) * prior) / ((1-sen) * prior + spc * (1-prior))

        return {
            "Budget": observation["Budget"],
            "Features": self.features,
            "Build_progress": observation["Build_progress"],
            "Test_progress": observation["Test_progress"]
        }

if __name__ == "__main__":
    args = parse_args()

    design_yaml_path = args.design
    design = edss.design_from_yaml(design_yaml_path)

    # Environment check
    env = edss.EDSSEnv(design)
    check_env(env, warn=True)
    env = edss.POMDPWrap(env)
    check_env(env, warn=True)
    env = edss.BayesWrap(env)
    check_env(env, warn=True)

    # Initialize Environmetns
    # Create & Train agents
    mdp_agent = None
    if not os.path.isfile(os.path.join(args.exp_dir, "mdp_agent.zip")) or args.retrain:
        mdp_env = make_vec_env(make_EDSSEnv(design), n_envs=args.num_envs, seed=args.seed)
        mdp_agent = PPO("MultiInputPolicy", mdp_env,
                    n_steps=args.num_steps, batch_size=args.batch_size, n_epochs=args.num_epochs,
                    ent_coef=args.ent_coef, gamma=args.gamma, seed=args.seed, verbose=1,
                    tensorboard_log=args.exp_dir,
                    device=args.device
                    )
        mdp_agent.learn(total_timesteps=args.training_time, tb_log_name="MDP PPO Agent")
        mdp_agent.save(os.path.join(args.exp_dir, "mdp_agent"))

    pomdp_agent = None
    if not os.path.isfile(os.path.join(args.exp_dir, "pomdp_agent.zip")) or args.retrain:
        pomdp_env = make_vec_env(make_EDSSEnv(design, filter="pomdp"), n_envs=args.num_envs, seed=args.seed)
        pomdp_agent = PPO("MultiInputPolicy", pomdp_env,
                    n_steps=args.num_steps, batch_size=args.batch_size, n_epochs=args.num_epochs,
                    ent_coef=args.ent_coef, gamma=args.gamma, seed=args.seed, verbose=1,
                    tensorboard_log=args.exp_dir,
                    device=args.device
                    )
        pomdp_agent.learn(total_timesteps=args.training_time, tb_log_name="POMDP Naive PPO Agent")
        pomdp_agent.save(os.path.join(args.exp_dir, "pomdp_agent"))

    bayes_agent = None
    if not os.path.isfile(os.path.join(args.exp_dir, "bayes_agent.zip")) or args.retrain:
        bayes_env = make_vec_env(make_EDSSEnv(design, filter="bayes", transform=args.bayes_transform), n_envs=args.num_envs, seed=args.seed)
        bayes_agent = PPO("MultiInputPolicy", bayes_env,
                    n_steps=args.num_steps, batch_size=args.batch_size, n_epochs=args.num_epochs,
                    ent_coef=args.ent_coef, gamma=args.gamma, seed=args.seed, verbose=1,
                    tensorboard_log=args.exp_dir,
                    device=args.device
                    )
        bayes_agent.learn(total_timesteps=args.training_time, tb_log_name="POMDP Bayes PPO Agent")
        bayes_agent.save(os.path.join(args.exp_dir, "bayes_agent"))

    recurrent_agent = None
    if not os.path.isfile(os.path.join(args.exp_dir, "recurrent_agent.zip")) or args.retrain:
        pomdp_recurrent_env = make_vec_env(make_EDSSEnv(design,filter="pomdp"), n_envs=args.num_envs, seed=args.seed)
        recurrent_agent = RecurrentPPO("MultiInputLstmPolicy", pomdp_recurrent_env, 
                    n_steps=args.num_steps, batch_size=args.batch_size, n_epochs=args.num_epochs,
                    ent_coef=args.ent_coef, gamma=args.gamma, seed=args.seed, verbose=1,
                    tensorboard_log=args.exp_dir,
                    device=args.device
                    )
        recurrent_agent.learn(total_timesteps=args.training_time, tb_log_name="POMDP Recurrent PPO Agent")
        recurrent_agent.save(os.path.join(args.exp_dir, "recurrent_agent"))

    if mdp_agent is None:
        mdp_agent = PPO.load(os.path.join(args.exp_dir, "mdp_agent"))
    if pomdp_agent is None:
        pomdp_agent = PPO.load(os.path.join(args.exp_dir, "pomdp_agent"))
    if bayes_agent is None:
        bayes_agent = PPO.load(os.path.join(args.exp_dir, "bayes_agent"))
    if recurrent_agent is None:
        recurrent_agent = RecurrentPPO.load(os.path.join(args.exp_dir, "recurrent_agent"))

    env = edss.EDSSEnv(design)
    bayes = BayesEstimator(env, transform=args.bayes_transform)
    obs, _ = env.reset()
    episode_starts = torch.ones((0,), dtype=bool)
    _, recurrent_states = recurrent_agent.predict({"Budget": obs["Budget"], "Build_progress": obs["Build_progress"], "Test_progress": obs["Test_progress"]}, episode_start=np.ones((1,),dtype=bool))
    recurrent_states = (torch.tensor(recurrent_states[0]), torch.tensor(recurrent_states[1]))
    done = False
    idx = ["MDP", "POMDP", "Bayes", "Recurrent"]
    columns = []
    for i in range(len(obs["Build_progress"])):
        columns.append(f'Build-{i+1}')
    for i in range(len(obs["Test_progress"])):
        columns.append(f'Test-{i+1}')
    columns.append('Terminate')

    with torch.no_grad():
        while not done:
            p_obs = {"Budget": obs["Budget"], "Build_progress": obs["Build_progress"], "Test_progress": obs["Test_progress"]}

            t_obs, _ = mdp_agent.policy.obs_to_tensor(obs)
            mdp_dist = mdp_agent.policy.get_distribution(t_obs).distribution.probs
            
            t_obs, _ = pomdp_agent.policy.obs_to_tensor(p_obs)
            pomdp_dist = pomdp_agent.policy.get_distribution(t_obs).distribution.probs

            t_obs, _ = bayes_agent.policy.obs_to_tensor(bayes.estimate(p_obs))
            bayes_dist = bayes_agent.policy.get_distribution(t_obs).distribution.probs

            t_obs, _ = recurrent_agent.policy.obs_to_tensor(p_obs)
            recurrent_dist, recurrent_states = recurrent_agent.policy.get_distribution(t_obs, recurrent_states, episode_starts)
            recurrent_dist = recurrent_dist.distribution.probs

            print(obs, bayes.estimate(p_obs)["Features"])
            df = pd.DataFrame([mdp_dist[0].numpy(), pomdp_dist[0].numpy(), bayes_dist[0].numpy(), recurrent_dist[0].numpy()], index=idx, columns=columns)
            #df.style.highlight_max(axis=1, props='font-weight:bold')
            s = df.style.highlight_max(axis=1)
            #print(df)
            display(df.style)
            print()

            action, _ = pomdp_agent.predict(p_obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        #mdp_action, _ = mdp_agent.predict(obs, deterministic=True)
        #pomdp_action, _ = pomdp_agent.predict(p_obs, deterministic=True)
        #recurrent_action, recurrent_states = recurrent_agent.predict(p_obs, deterministic=True)

raise SystemExit
