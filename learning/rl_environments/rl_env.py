from tensorforce import TensorforceError, util
import math
from tensorforce.environments import Environment
import time
from rlbot.utils import public_utils, logging_utils
import sys
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\bot-1\\src')
from util.vec import Vec3
from util.orientation import Orientation
from .game_values import InputOptions, OutputOptions

# Environment for Rocket League training
class RLEnvironment(Environment):
	def __init__(self, max_time=100, name='bot', frames_per_sec = 10, message_throttle = 1, input_exclude=[], output_exclude=[]):
		super(Environment, self).__init__()
		self.name = name
		self.input_exclude = input_exclude
		self.output_exclude = output_exclude
		self.ep = 0
		self.max_time = max_time
		self.max_timesteps = round(max_time * frames_per_sec)
		self.frames_per_sec = frames_per_sec
		self.fail = False
		self.rl_state = {}
		self.ball_velocity = 0
		self.comms = None
		self.timestep = 0
		self.max_ball_v = 0
		self.ep_reward = 0
		self.max_ep_reward = 0
		self.agent = None
		self.counter = 0
		self.message_throttle = message_throttle

	@staticmethod
	def get_max_timesteps(max_time, frames_per_sec):
		return round(max_time * frames_per_sec)

	def setComms(self, comms):
		self.comms = comms

	def setRLState(self, game_tick):
		my_car = game_tick.game_cars[0]
		car_location = Vec3(my_car.physics.location)
		car_rotation = Orientation(my_car.physics.rotation)
		car_velocity = Vec3(my_car.physics.velocity)
		ball_location = Vec3(game_tick.game_ball.physics.location)
		ball_velocity = (
			game_tick.game_ball.physics.velocity.x +
			game_tick.game_ball.physics.velocity.y +
			game_tick.game_ball.physics.velocity.z
		)
		dist = car_location.dist(ball_location)


		self.ball_velocity = ball_velocity

		self.rl_state = {
			InputOptions.BALL_DISTANCE: [dist],
			InputOptions.CAR_ORIENTATION: [
				car_rotation.up.as_arr(), 
				car_rotation.right.as_arr(),
				car_rotation.forward.as_arr()
			],
			InputOptions.CAR_HEIGHT: [car_location.z],
			InputOptions.CAR_VELOCITY: car_velocity.as_arr(),
			InputOptions.JUMPED: [my_car.jumped],
			InputOptions.DOUBLE_JUMPED: [my_car.double_jumped],
			InputOptions.BALL_POSITION: ball_location.as_arr(),
			InputOptions.CAR_POSITION: car_location.as_arr()
		}
		return self.rl_state

	def states(self):
		# Ball distance
		# Time in air
		all_state_options = {
			InputOptions.BALL_DISTANCE: {
				"type": "float",
				"shape": 1,
				"min_value": -13272,
				"max_value": 13272,
			},
			InputOptions.CAR_ORIENTATION: {
				"type": "float",
				"shape": (3, 3),
				"min_value": -1,
				"max_value": 1,
			},
			InputOptions.CAR_HEIGHT: {
				"type": "float",
				"shape": 1,
				"min_value": 0,
				"max_value": 2045,
			},
			InputOptions.CAR_VELOCITY: {
				"type": "float",
				"shape": 3,
				"min_value": -2301,
				"max_value": 2301,
			},
			InputOptions.JUMPED: {
				"type": "bool",
				"shape": 1,
			},
			InputOptions.DOUBLE_JUMPED: {
				"type": "bool",
				"shape": 1,
			},
			InputOptions.CAR_POSITION: {
				"type": "float",
				"shape": 3,
				"min_value": -5120 - 881,
				"max_value": 5120 + 881,
			},
			InputOptions.BALL_POSITION: {
				"type": "float",
				"shape": 3,
				"min_value": -5120 - 881,
				"max_value": 5120 + 881,
			}
		}
		return { key:val for key, val in all_state_options.items() if key not in self.input_exclude }

	def actions(self):
		return {
			OutputOptions.JUMP: {
				"type": "bool",
				"shape": 1,
			},
			OutputOptions.PITCH: {
				"type": "int",
				"shape": 1,
				"num_values": 3,
			}
		}

	def set_agent(self, agent):
		self.agent = agent

	def save_agent(self):
		self.agent.save(directory='best/{0}'.format(self.name), append='episodes')

	def reset(self):
		logging_utils.log_warn('.'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('Reward: {0:.2f}'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})
		logging_utils.log_warn('---------'.format(self.ep_reward), {})

		if self.agent and self.ep_reward > self.max_ep_reward and self.ep > 2:
			self.save_agent()
			self.max_ep_reward = self.ep_reward

		self.timestep = 0
		self.ep += 1
		self.fail = True
		self.ball_velocity = 0
		self.rl_state = {}
		self.ep_reward = 0
		self.comms and self.comms.outgoing_broadcast.empty()
		return {
			InputOptions.BALL_DISTANCE: [0],
			InputOptions.BALL_POSITION: [0, 9, 9],
			InputOptions.CAR_POSITION: [0, 0, 0],
			InputOptions.CAR_ORIENTATION: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
			InputOptions.CAR_HEIGHT: [0],
			InputOptions.CAR_VELOCITY: [0, 0, 0],
			InputOptions.JUMPED: [False],
			InputOptions.DOUBLE_JUMPED: [False]
		}

	def execute(self, actions):
		self.timestep += 1

		if (self.comms is not None):
			serializable_actions = {key:val.tolist() for key, val in actions.items()}
			self.comms.outgoing_broadcast.put(serializable_actions)

		self.throttled_log('Ep #{0}--------T:{1}'.format(self.ep, self.timestep))
		reward = self.reward(self.rl_state)
		self.ep_reward += reward
		self.throttled_log("Reward: {0:.2f}".format(self.ep_reward))

		# State, terminal, reward
		return self.rl_state, 1 if self.timestep >= self.max_timesteps else 0, reward

	def reward(self):
		raise NotImplementedError

	def episode_reward(self, parallel=None):
		return None

	def throttled_log(self, message):
		if self.timestep % self.message_throttle == 0:
			logging_utils.log_warn(message, {})

	def update_throttle(self, game_tick):
		kickoff_pause = game_tick.game_info.is_kickoff_pause
		ticks_per_sec = 46.66
		frame_throttle = round(ticks_per_sec / self.frames_per_sec)


		if not kickoff_pause:
			if self.counter == frame_throttle:
				self.counter = 0
				return True
			self.counter += 1

		return False

