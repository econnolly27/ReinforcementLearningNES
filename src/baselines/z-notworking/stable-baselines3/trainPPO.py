import gym, os
import numpy as np
from gym.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from setup_env import setup_env, MarioDiscretizerSimple, MarioDiscretizerComplex, FrameStack
from torchsummary import summary
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

TENSOR_BOARD = os.path.dirname(os.path.abspath(__file__))

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = setup_env('SMB-JU', 'Level1-1')
        # env = MarioDiscretizerSimple(env)
        env = MarioDiscretizerComplex(env)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=(84,84))
        # env = Monitor(env, TENSOR_BOARD+'/logs/'+rank+'/')
        # env = FrameStack(env, 4)
        env.seed(seed + rank)
        # check_env(env)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    # eval_env = setup_env('SMB-JU', 'Level1-1')
    # # env = MarioDiscretizerSimple(env)
    # eval_env = MarioDiscretizerComplex(eval_env)
    # eval_env = GrayScaleObservation(eval_env, keep_dim=True)
    # eval_env = ResizeObservation(eval_env, shape=(84,84))
    # # eval_env = VecTransposeImage(DummyVecEnv([eval_env]))

    # num_cpu = 1  # Number of processes to use
    # # Create the vectorized environment
    # temp = [make_env(i) for i in range(num_cpu)]
    # eval_env = SubprocVecEnv(temp)

    callback = CheckpointCallback(save_freq=100000, save_path=TENSOR_BOARD+'/logs/', name_prefix='PPO_model')

    # callback = EvalCallback(eval_env, best_model_save_path=TENSOR_BOARD+'/logs/', log_path=TENSOR_BOARD+'/logs/', eval_freq=10000, deterministic=True, render=True)

    num_cpu = 12  # Number of processes to use
    # Create the vectorized environment
    temp = [make_env(i) for i in range(num_cpu)]
    env = SubprocVecEnv(temp)

    # Instantiate the agent
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=TENSOR_BOARD+"/runs/", create_eval_env=True)
    # Train the agent
    model.learn(total_timesteps=int(20e6), callback=callback)

    model.save(TENSOR_BOARD+"/PPO_mario_final")
    # del model  # delete trained model to demonstrate loading

    # # Load the trained agent
    # env_eval = setup_env('SMB-JU', 'Level1-1')
    # env_eval = MarioDiscretizerComplex(env_eval)
    # env_eval = GrayScaleObservation(env_eval, keep_dim=True)
    # env_eval = ResizeObservation(env_eval, shape=(84,84))
    # model = PPO.load(TENSOR_BOARD+"/PPO_mario", env=env_eval)

    # # Evaluate the agent
    # # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=False)
    # # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # # Enjoy trained agent
    # # env = setup_env('SMB-JU', 'Level1-1')
    # # env = MarioDiscretizerComplex(env)
    # # env = GrayScaleObservation(env, keep_dim=True)
    # # env = ResizeObservation(env, shape=(84,84))
    # # obs = env.reset()
    # # for _ in range(10000):
    # #     action, _states = model.predict(obs)
    # #     obs, rewards, dones, info = env.step(action)
    # #     env.render()