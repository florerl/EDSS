import random, schema, yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper, Wrapper
from gymnasium.envs.registration import EnvSpec
from collections import defaultdict, namedtuple

# Schema for Engineering Design Program to simulate syntheis of
# List of discrete potential features with build and testing actions
## (Optional) Feature name.
## List of Build actions (mutually exclusive progress)
### Cost from budget consumed per synthesis cycle
### Minimum number of synthesis cycles to complete
### Average number of synthesis cycles to complete
## List of Test actions
### Cost from budget consumed for test
### Specificity of test
### Sensitivity of test
# Total budget for project at start
# Profit Elements: Key value for features i and j (in order above) is ( 2^i + 2^j )
#
# Assumption: Average > Mimimum
# Assumption: Build progress is gemetrically distributed, so transition probability
#   p = (Average - Minimum) 
# TODO: Feature prerequisites (featues a & b before c)
# TODO: Simplify specification of profit elements
# TODO: Non-geometric

Action = namedtuple('Action', ['type', 'cost', 'feature', 'prog_idx', 'attr1', 'attr2'])

# Engineering Design Synthesis Simulator
class EDSSEnv(gym.Env):
    metadata = {'render.modes': ['None']}
    spec = EnvSpec("EDSSEnv-v0")

    design_schema = schema.Schema({
        "features": [{
            schema.Optional("name"): str, 
            "build_actions": [{
                "cost": schema.And(int, lambda n: n >= 0),
                "minWork": schema.And(int, lambda n: n >= 0),
                "aveWork": schema.And(int, lambda n: n > 0)
            }],
            "test_actions": [{
                "cost": schema.And(int, lambda n: n >= 0),
                "specificity": schema.And(schema.Use(float), lambda n: 0.0 <= n <= 1.0),
                "sensitivity": schema.And(schema.Use(float), lambda n: 0.0 <= n <= 1.0)
            }]
        }],
        "budget": schema.And(int, lambda n: n > 0),
        "profit_elems": {
            int: schema.Or(int,float)
        }
    })

    def __init__(self, design, initData=None):
        super(EDSSEnv, self).__init__()
        assert initData is None
        self.initData = initData

        # Import design to syhnthesize
        try:
            designData = self.design_schema.validate(design)
        except schema.SchemaError as se:
            raise se
        self.info = designData

        # Internal State Construction
        features = designData['features']
        self.actions: list[Action] = []
        buildActions: list[Action] = []
        testActions: list[Action] = []
        b_i = t_i = 0
        b_work_min = 0
        for i, f in enumerate(features):
            for b in f['build_actions']:
                assert b['minWork'] < b['aveWork']
                attr2 = 1.0/(b['aveWork']-b['minWork'])
                work_min = -b['minWork']*attr2
                buildActions.append(Action('build', b['cost'], i, b_i, b['minWork'], attr2))
                b_work_min = work_min if work_min < b_work_min else b_work_min
                b_i += 1
            for t in f['test_actions']:
                testActions.append(Action('test', t['cost'], i, t_i, t['specificity'], t['sensitivity']))
                t_i += 1
        for b in buildActions:
            self.actions.append(b)
        for t in testActions:
            self.actions.append(t)
        self.actions.append(Action('done', 0, -1, -1, 0, 0))
        self.profit_function = defaultdict(int)
        for k,v in designData['profit_elems'].items():
            self.profit_function[k] = v

        # Observation Space
        self.observation_space = spaces.Dict({
            "Budget": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int_),
            "Features": spaces.MultiBinary(len(features)),
            "Build_progress": spaces.Box(low= b_work_min, high= np.inf, shape=(len(buildActions),), dtype=np.single),
            "Test_progress": spaces.Box(low=-1, high=1, shape=(len(testActions),), dtype=np.int_)
            })
        # Action Space
        self.action_space = spaces.Discrete(len(self.actions))

        # Initialization
        if self.initData is None:
            self.features = np.zeros(len(features), dtype=bool)
            self.budget = self.info['budget']
            self.build_progress = np.zeros(len(buildActions), dtype=int)
            self.test_progress = np.full(len(testActions), -1)
            self.build_shift = np.empty_like(self.build_progress, dtype=np.single)
            self.build_scale = np.empty_like(self.build_progress, dtype=np.single)
            for a in self.actions:
                if a.type == "build":
                    self.build_shift[a.prog_idx] = -a.attr1
                    self.build_scale[a.prog_idx] = a.attr2


    def _get_obs(self):
        return {
            "Budget": np.asarray([self.budget], dtype=np.int_), 
            "Features": self.features.astype(np.int8),
            "Build_progress": np.float32((self.build_progress + self.build_shift) * self.build_scale), 
            "Test_progress": self.test_progress
            }
    
    def _get_info(self):
        return self.info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.initData is None:
           self.features = np.zeros_like(self.features, dtype=bool)
           self.budget = self.info['budget']
           self.build_progress = np.zeros_like(self.build_progress)
           self.test_progress = np.full_like(self.test_progress, -1)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        terminated = False
        reward = 0

        #do action
        # TODO: Revisit default behavior when not enough budget for chosen action
        action = self.actions[action]
        if action.cost >= self.budget:
            action = self.actions[-1]
        self.budget -= action.cost
        match action.type:
            case "build":
                for a in self.actions:
                    if a.type == "test" and a.feature == action.feature:
                        self.test_progress[a.prog_idx] = -1
                self.build_progress[action.prog_idx] += 1
                if self.features[action.feature] or self.build_progress[action.prog_idx] < action.attr1:
                    pass
                else:
                    self.features[action.feature] = random.random() < action.attr2
            case "test":
                if self.test_progress[action.prog_idx] != -1:
                    pass
                elif self.features[action.feature]:
                    self.test_progress[action.prog_idx] = int(random.random() < action.attr2)
                else:
                    self.test_progress[action.prog_idx] = int(random.random() > action.attr1)
            case "done":
                terminated = True
                reward = self._profit() + self.budget
            case _:
                pass

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def _profit(self):
        features_idx = self.features.dot(1 << np.arange(self.features.size)[::1])
        return self.profit_function[features_idx]
    
def design_from_yaml(path: str):
    with open(path) as f:
        design = yaml.safe_load(f)
    return design

class POMDPWrap(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "Budget": env.observation_space["Budget"],
            "Build_progress": env.observation_space["Build_progress"],
            "Test_progress": env.observation_space["Test_progress"]
            })
    
    def observation(self, observation):
        return {
            "Budget": observation["Budget"],
            "Build_progress": observation["Build_progress"],
            "Test_progress": observation["Test_progress"]
        }
    
class BayesWrap(ObservationWrapper):
    def __init__(self, env: gym.Env, transform=None):
        super().__init__(env)
        self.buildActions: list[Action] = []
        self.testActions: list[Action] = []
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
        self.observation_space = spaces.Dict({
            "Budget": env.observation_space["Budget"],
            "Features": spaces.Box(low=0.0, high=1.0, shape=(len(env.features),), dtype=np.single),
            "Build_progress": env.observation_space["Build_progress"],
            "Test_progress": env.observation_space["Test_progress"]
            })
    
    def observation(self, observation):
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
