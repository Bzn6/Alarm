import datetime
import gymnasium 
import numpy as np
import uuid

'''
minigrid 환경을 DreamerV3에서 사용할 수 있도록 래핑하는 파일
'''

class TimeLimit(gymnasium.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration: # 시간 초과 시 truncated=True 로 처리
            truncated = True
            if "discount" not in info: 
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None 
        done = terminated or truncated # gym에서는 done이 terminated와 truncated의 논리합으로 처리됨 (dreamerv3에서는 gym 환경을 사용하므로 done 변수로 표현해야 함)
        return obs, reward, done, info
    
    def reset(self):
        self._step = 0 
        obs, info = self.env.reset() ## 환경을 리셋하고, obs를 받아옴(gymnasium에서는 reset이 obs와 info를 반환)
        return obs
    
    
class OneHotAction(gymnasium.Wrapper):
    """
    이산(Discrete) action set을 위한 class
    """
    def __init__(self, env):
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gymnasium.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        obs, info = self.env.reset() # 환경을 리셋하고, obs를 받아옴(gymnasium에서는 reset이 obs와 info를 반환)
        return obs

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = dict(self.env.observation_space.spaces)
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gymnasium.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["obs_reward"] = np.array([reward], dtype=np.float32)
        done = terminated or truncated # gym에서는 done이 terminated와 truncated의 논리합으로 처리됨 (dreamerv3에서는 gym 환경을 사용하므로 done 변수로 표현해야 함)
        return obs, reward, done, info
    
    def reset(self):
        obs, info = self.env.reset() # 환경을 리셋하고, obs를 받아옴(gymnasium에서는 reset이 obs와 info를 반환)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gymnasium.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        obs, info = self.env.reset()
        return obs
