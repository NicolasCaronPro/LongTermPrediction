import gym
from gym import spaces
import numpy as np
from tianshou.env import DummyVectorEnv

class PixelEnv(gym.Env):
    def __init__(self, features_block, next_true_states, valid_transitions_dict):
        super().__init__()

        # Chaque environnement ne gère qu'un seul pixel / échantillon
        self.features = features_block  # shape: (F, T)
        # État courant : dernier feature du dernier timestep
        self.current_states = features_block[-1, -1]
        self.true_next_states = next_true_states        # shape: (B,)
        self.valid_transitions = valid_transitions_dict
        self.batch_size = self.features.shape[0]

        # Transitions valides pixel par pixel
        self.possible_next_states = valid_transitions_dict[int(self.current_states)]

        # Action = un entier par pixel
        self.action_space = spaces.Discrete(len(self.possible_next_states))

        # Observation = (B, F+1, T)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.features.shape),
            dtype=np.float32
        )

        self.done = False

    def reset(self):
        self.done = False

        return self.features
    
    def step(self, action):
        if self.done:
            raise Exception("reset() avant de recommencer")

        predicted = self.possible_next_states[action]

        rewards = float(predicted == self.true_next_states)
        self.done = True
        return self.features, rewards, self.done, {}
    

def create_vec_env_from_block(features_block, next_states, valid_transitions):
    """Crée un ``DummyVectorEnv`` Tianshou à partir d'un bloc ``(B, F, T)``."""
    B = features_block.shape[0]
    envs = []

    for i in range(B):
        envs.append(lambda i=i: PixelEnv(
            features_block[i],
            next_true_states=next_states[i],
            valid_transitions_dict=valid_transitions
        ))

    return DummyVectorEnv(envs)
