import gym
import numpy as np
from dm_control import suite # DMC 환경을 불러오기 위한 모듈 

class DeepMindControl:
    '''
    DMC 환경을 DreamerV3에서 사용할 수 있도록 래핑하는 class.
    '''
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=0, seed=0):
        '''
        name : 환경 이름(예: 'dmc_walker_walk')
        action_repeat : 행동을 몇 프레임 동안 유지할지 
        size : 입력 이미지의 해상도 (예: (64, 64))  
        camera : 사용할 카메라 ID (기본값은 None, 그러나 hopper, walker는 0 이므로 0으로 설정)
        seed : 환경의 랜덤 시드 (기본값은 0)
        '''
        domain, task = name.split("_", 1) # "_"를 기준으로 도메인과 태스크를 분리 (예: 'dmc_walker_walk' -> 'walker', 'walk')
        self._env = suite.load( # DMC 환경을 불러오는 함수
                domain, # 도메인 이름 (예: 'walker') - Line 19에서 분리된 부분
                task, # 태스크 이름 (예: 'walk') - Line 19에서 분리된 부분
                task_kwargs={"random": seed}, # 태스크에 랜덤 시드를 설정 
            )
        
        self._action_repeat = action_repeat #__init 메서드에서 받은 action_repeat 값을 저장
        self._size = size #__init 메서드에서 받은 size 값을 저장 
        self.reward_range = [-np.inf, np.inf] # reward의 범위를 무한대로 설정  

    @property
    def observation_space(self):
        '''
        DMC 환경의 observation space를 DreamerV3가 이해 할 수 있도록 래핑하는 함수.
        '''
        spaces = {} # DMC 환경의 관측 공간을 저장할 딕셔너리 생성 
        for key, value in self._env.observation_spec().items(): 
            # Line 20에서 정의한 _env를 observation_spec() 메서드를 통해 관측 공간을 가져옴
            # Key는 관측 공간의 각 요소를 나타냄(예: 관절 위치, 속도, 방향 등)
            # value는 각 key에 대한 관측 값의 data 스펙을 나타냄 
            if len(value.shape) == 0: # value가 스칼라 값인 경우
                shape = (1,) # 스칼라 값을 1차원 배열로 변환 
            else: # value가 벡터나 행렬인 경우
                shape = value.shape # 원래 shape를 유지 
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32) # Line 35의 space dictionary에 key와 value를 추가 (예: key-관절 위치, 속도 등/ value-spaces.Box 객체로 정의한 shape 형태의 data를 갖고 shape의 각 원소 값들은 무한대 범위의 실수값을 가짐)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8) # Line 35의 space dictionary에 "image" key를 추가 (이미지 관측 공간을 정의, 0-255 범위의 RGB 이미지로 크기는 _size + (3,) - 여기서 3은 RGB 채널 수를 나타냄)
        return gym.spaces.Dict(spaces) # 관측 공간을 gym.spaces.Dict 형태로 반환

    @property
    def action_space(self):
        '''
        DMC 환경의 action space를 DreamerV3가 이해 할 수 있도록 래핑하는 함수.
        '''
        spec = self._env.action_spec() # Line 20에서 정의한 _env를 action_spec() 메서드를 통해 액션 공간을 가져옴
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32) # 액션 공간을 gym.spaces.Box 형태로 반환 (spec.minimum과 spec.maximum은 액션의 최소값과 최대값을 나타냄)

    def step(self, action):
        '''
        DMC 환경에서 action을 선택 후, 스텝을 실행한 결과를
        DreamerV3가 사용할 수 있는 형태(obs, reward, done, info)로 래핑하는 함수.
        '''
        assert np.isfinite(action).all(), action # action이 모두 유한한 값인지 확인 (assert문 구조 : assert condition, error_message_or_object : 만약 condition이 False라면 error_message_or_object를 출력) 
        reward = 0 # 보상을 초기화 (reward는 누적 보상으로 사용됨) 
        for _ in range(self._action_repeat): #__init 메서드에서 받은 action_repeat 만큼 반복 (그러므로...1회만 반복하는 루프임)
            time_step = self._env.step(action) # DMC 환경 객체인 _env의 step 메서드를 호출하여 action을 실행하고 time_step을 받아옴 - time_step에는...time_step.observation, time_step.reward, time_step.discount, time_step.last()가 있음 (이 함수 step이 아님 주의!) 
            reward += time_step.reward or 0 # reward를 누적 (time_step.reward가 None인 경우 0으로 처리)
            if time_step.last(): # time_step이 마지막 스텝인 경우 중단
                break
              
        obs = dict(time_step.observation) # time_step에서 관측(observation)을 딕셔너리 형태로 추출 
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()} # 스칼라 관측값은 1D 배열로 변환하고, 나머지는 그대로 유지
        obs["image"] = self.render() # 이미지 관측을 추가 (render 메서드를 통해 현재 상태의 이미지를 가져옴) 
        # There is no terminal state in DMC (그러므로 discount == 0 이면 마지막 스텝이므로 is_terminal을 True로 설정)
        obs["is_terminal"] = Fase if time_step.first() else time_step.discount == 0 # 마지막 스텝이 아니면 False, 첫 번째 스텝이면 True    
        obs["is_first"] = time_step.first() # 첫 번째 스텝인지 여부를 나타내는 플래그
        done = time_step.last() # 마지막 스텝인지 여부를 나타내는 플래그 
        info = {"discount": np.array(time_step.discount, np.float32)} # DMC 환경의 discount 값을 info에 저장 (numpy float32 형태로 변환 - 1 또는 0이 전달 됨)
        return obs, reward, done, info # DreamerV3가 사용할 수 있는 형태로 반환 (obs, reward, done, info)

    def reset(self):
        '''
        Dreamer에서 에피소드 시작 시 호출되는 함수 
        DMC 환경을 초기화하고, 첫 관측(observation)을 반환
        '''
        time_step = self._env.reset() # 첫 번째 step 정보(time_step)**를 가져옴
        obs = dict(time_step.observation) # time_step에서 관측(observation)을 딕셔너리 형태로 추출 
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()} # 스칼라 관측값은 1D 배열로 변환하고, 나머지는 그대로 유지 
        obs["image"] = self.render() # 이미지 관측을 추가 (render 메서드를 통해 현재 상태의 이미지를 가져옴) 
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0  # 마지막 스텝이 아니면 False, 첫 번째 스텝이면 True  
        obs["is_first"] = time_step.first() # 첫 번째 스텝인지 여부를 나타내는 플래그
        return obs

    def render(self, *args, **kwargs):
        '''
        현재 환경 상태를 시각적으로 출력하는 렌더링 함수
        '''
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
