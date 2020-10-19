import os
import gym
from gym.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from setup_env import setup_env, MarioDiscretizerSimple, MarioDiscretizerComplex

TENSOR_BOARD = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    env_eval = setup_env('SMB-JU', 'Level1-1')
    env_eval = MarioDiscretizerComplex(env_eval)
    env_eval = GrayScaleObservation(env_eval, keep_dim=True)
    env_eval = ResizeObservation(env_eval, shape=(84,84))
    model = PPO.load(TENSOR_BOARD+"/PPO_model_8999995_steps", env=env_eval)

    # Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, render=False)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    print('Trial trained agent... Ctrl+C to stop')
    obs = env_eval.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env_eval.step(action)
        env_eval.render()
        if done:
            obs = env_eval.reset()

        