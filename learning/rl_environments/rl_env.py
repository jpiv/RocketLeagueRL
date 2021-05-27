from tensorforce import TensorforceError, util
import math
from tensorforce.environments import Environment
import time
from rlbot.utils import public_utils, logging_utils

from pathlib import Path
import os
import sys
import time
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\rocket-league-gym')
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\RLBot\\src\\main\\python')
import rlgym_local as rlgym
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\bot-1\\src')
from util.vec import Vec3
from util.orientation import Orientation, OrientationGym, relative_location
from .game_values import InputOptions, OutputOptions, GameValues
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\RLBotTraining')
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\RLBot\\src\\main\\python')
from rlbottraining_local.exercise_runner import run_module
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\learning\\training')
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\FieldSet')
from field_set import FieldSet
import custom_training
from rlbot.utils.logging_utils import get_logger, DEFAULT_LOGGER
import threading
from rlgym_local.communication import CommunicationHandler, Message
from rlgym_local.gamelaunch import launch_rocket_league
from rlbot_local.gamelaunch.epic_launch import launch_with_epic_simple
from rlbot_local.setup_manager import setup_manager_context, ROCKET_LEAGUE_PROCESS_INFO
from rlbot_local import gateway_util

# Environment for Rocket League training
class RLEnvironment(Environment):
	def __init__(self, max_time=100, name='bot', frames_per_sec = 10, message_throttle = 1, input_exclude=[], output_exclude=[], no_launch=False):
		super(Environment, self).__init__()
		self.name = name
		self.input_exclude = input_exclude
		self.output_exclude = output_exclude
		self.ep = 0
		self.max_time = max_time
		self.max_timesteps = round(max_time * frames_per_sec)
		self.frames_per_sec = frames_per_sec
		self.fail = False
		self.rl_state = None 
		self.game_tick = None 
		self.comms = None
		self.timestep = 0
		self.ep_reward = 0
		self.max_ep_reward = 0
		self.agent = None
		self.counter = 0
		self.should_execute = False
		self.executed = False
		self.message_throttle = message_throttle
		self.pipe_comms = None
		self.logger = get_logger(DEFAULT_LOGGER)
		if not no_launch:
			comms = CommunicationHandler()
			self.logger.warning(os.getpid())
			pipe_id = comms.format_pipe_id(os.getpid()) + '{name}'.format(name=name)
			self.logger.warning(pipe_id)
			

			from rlgym_local.gamelaunch import launch_rocket_league

			from rlgym_local.envs import match_factory
			rl_gym_match_config = dict(
				self_play=False,
				tick_skip=14,
				random_resets=None,
				reward_fn = None,
				obs_builder = None,
				ep_len_minutes=2000,
				team_size=1,
				spawn_opponents=False,
				game_speed=2,	
				terminal_conditions=None,
			)
			default_port = 23233
			ideal_args_rlbot = ROCKET_LEAGUE_PROCESS_INFO.get_ideal_args(default_port)
			ideal_args_rlgym = ['-pipe', f'{pipe_id}', '-nomovie']

			launch_with_epic_simple(ideal_args_rlbot + ideal_args_rlgym)

			self.logger.warning('Starting Field Set...')
			self.field_set = FieldSet()
			self.logger.warning('Field set connected.')
			
			# def start_module():
			self.logger.warning('Opening pipe...')
			comms.open_pipe(self.logger, pipe_id)
			self.pipe_comms = comms
			self.logger.warning('Pipe connected.')


			# t1 = threading.Thread(target=start_module)
			# t1.start()

			# time.sleep(10)
			# run_module(
			# 	custom_training,
			# 	self,
			# 	pipe_id
			# )


			self.match = match_factory.build_match('default', **rl_gym_match_config)
			comms.send_message(header=Message.RLGYM_CONFIG_MESSAGE_HEADER, body=self.match.get_config())
			self.logger.warning(pipe_id)


	@staticmethod
	def get_max_timesteps(max_time, frames_per_sec):
		return round(max_time * frames_per_sec)

	def setComms(self, comms):
		self.comms = comms

	def filter_states(self, states):
		return { key:val for key, val in states.items() if key not in self.input_exclude }

	def game_tick_packet_to_rl_state(self, game_tick):
		my_car = game_tick.game_cars[0]
		car_location = Vec3(my_car.physics.location)
		car_rotation = Orientation(my_car.physics.rotation)
		car_velocity = Vec3(my_car.physics.velocity)
		ball_location = Vec3(game_tick.game_ball.physics.location)
		ball_location_rel = relative_location(car_location, car_rotation, ball_location)
		goal_location = Vec3(0, GameValues.GOAL_CENTER_Y, GameValues.GOAL_CENTER_Z)
		car_location_rel = relative_location(goal_location, car_rotation, car_location)
		dist = car_location.dist(ball_location)

	def gym_to_rl_state(self, state):
		player = state.players[0]
		my_car = player.car_data
		ball = state.ball
		# car_location = Vec3(my_car.physics.location)
		car_location = Vec3(*my_car.position)
		car_rotation = OrientationGym(my_car)
		car_velocity = Vec3(*my_car.linear_velocity)
		ball_location = Vec3(*ball.position)
		ball_location_rel = relative_location(car_location, car_rotation, ball_location)
		goal_location = Vec3(0, GameValues.GOAL_CENTER_Y, GameValues.GOAL_CENTER_Z)
		car_location_rel = relative_location(goal_location, car_rotation, car_location)
		dist = car_location.dist(ball_location)
		self.game_tick = state
		# self.should_execute = self.update_throttle(state) or not self.executed


		self.rl_state = {
			InputOptions.BALL_DISTANCE: [dist],
			InputOptions.CAR_ORIENTATION: [
				car_rotation.up.as_arr(), 
				car_rotation.right.as_arr(),
				car_rotation.forward.as_arr()
			],
			InputOptions.CAR_HEIGHT: [car_location.z],
			InputOptions.CAR_VELOCITY: car_velocity.as_arr(),
			InputOptions.JUMPED: [player.on_ground],
			InputOptions.DOUBLE_JUMPED: [player.has_flip],
			InputOptions.BALL_POSITION: ball_location.as_arr(),
			InputOptions.CAR_POSITION: car_location.as_arr(),
			InputOptions.BALL_POSITION_REL: ball_location_rel.as_arr(),
			InputOptions.BALL_DIRECTION: ball_location_rel.normalized().as_arr(),
			InputOptions.CAR_POSITION_REL: car_location_rel.as_arr(),
			InputOptions.CAR_VELOCITY_MAG: [car_velocity.length()],
		}
		return self.filter_states(self.rl_state)

	def states(self):
		# Ball distance
		# Time in air
		all_state_options = {
			InputOptions.BALL_DISTANCE: {
				"type": "float",
				"shape": 1,
				"min_value": -GameValues.FIELD_MAX_DISTANCE,
				"max_value": GameValues.FIELD_MAX_DISTANCE,
			},
			InputOptions.CAR_ORIENTATION: {
				"type": "float",
				"shape": (3, 3),
				"min_value": -1.0001,
				"max_value": 1.0001,
			},
			InputOptions.CAR_HEIGHT: {
				"type": "float",
				"shape": 1,
				"min_value": 0,
				"max_value": GameValues.FIELD_HEIGHT,
			},
			InputOptions.CAR_VELOCITY: {
				"type": "float",
				"shape": 3,
				"min_value": -GameValues.CAR_MAX_VELOCITY,
				"max_value": GameValues.CAR_MAX_VELOCITY,
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
				"min_value": GameValues.FIELD_MIN_Y,
				"max_value": GameValues.FIELD_MAX_Y,
			},
			InputOptions.BALL_POSITION: {
				"type": "float",
				"shape": 3,
				"min_value": GameValues.FIELD_MIN_Y,
				"max_value": GameValues.FIELD_MAX_Y,
			},
			InputOptions.BALL_POSITION_REL: {
				"type": "float",
				"shape": 3,
				"min_value": -GameValues.FIELD_MAX_DISTANCE,
				"max_value": GameValues.FIELD_MAX_DISTANCE,
			},
			InputOptions.BALL_DIRECTION: {
				"type": "float",
				"shape": 3,
				"min_value": -1,
				"max_value": 1,
			},
			InputOptions.CAR_POSITION_REL: {
				"type": "float",
				"shape": 3,
				"min_value": -GameValues.FIELD_MAX_DISTANCE,
				"max_value": GameValues.FIELD_MAX_DISTANCE,
			},
			InputOptions.CAR_VELOCITY_MAG: {
				"type": "float",
				"shape": 1,
				"min_value": -GameValues.CAR_MAX_VELOCITY,
				"max_value": GameValues.CAR_MAX_VELOCITY,
			},
		}
		return self.filter_states(all_state_options)

	def actions(self):
		all_actions = {
			OutputOptions.JUMP: {
				"type": "bool",
				"shape": 1,
			},
			OutputOptions.PITCH: {
				"type": "int",
				"shape": 1,
				"num_values": 3,
			},
			OutputOptions.ROLL: {
				"type": "int",
				"shape": 1,
				"num_values": 3,
			},
			OutputOptions.STEER: {
				"type": "int",
				"shape": 1,
				"num_values": 3,
			},
			OutputOptions.BOOST: {
				"type": "bool",
				"shape": 1,
			},
			OutputOptions.THROTTLE: {
				"type": "int",
				"shape": 1,
				"num_values": 3,
			},
			OutputOptions.E_BRAKE: {
				"type": "bool",
				"shape": 1,
			},
		}
		return {key:val for key, val in all_actions.items() if key not in self.output_exclude}

	def set_agent(self, agent):
		self.agent = agent

	def save_agent(self):
		self.agent.save(directory='best/{0}'.format(self.name), append='episodes')

	def match_reset(self):
		self.pipe_comms.send_message(header=Message.RLGYM_CONFIG_MESSAGE_HEADER, body=self.match.get_config())
		exception = self.pipe_comms.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER, body=Message.RLGYM_NULL_MESSAGE_BODY)
		if exception:
			self.logger.warning(exception)
			raise Exception('error')
		message, _ = self.pipe_comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)
		state = self.match.parse_state(message.body)
		self.logger.warning(state)
		self.match.episode_reset(state)
		return self.gym_to_rl_state(state)

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
		self.game_tick = None 
		self.rl_state = None 
		self.ep_reward = 0
		self.executed = False
		self.should_execute = False
		self.comms and self.comms.outgoing_broadcast.empty()
		all_state_options = self.match_reset()
		self.field_set.randomize_ball_state()
		# all_state_options = {
		# 	InputOptions.BALL_DISTANCE: [0],
		# 	InputOptions.BALL_POSITION: [0, 0, 0],
		# 	InputOptions.CAR_POSITION: [0, 0, 0],
		# 	InputOptions.CAR_ORIENTATION: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
		# 	InputOptions.CAR_HEIGHT: [0],
		# 	InputOptions.CAR_VELOCITY: [0, 0, 0],
		# 	InputOptions.JUMPED: [False],
		# 	InputOptions.DOUBLE_JUMPED: [False],
		# 	InputOptions.BALL_POSITION_REL: [0, 500, 0],
		# 	InputOptions.BALL_DIRECTION: [0, 1, 0],
		# 	InputOptions.CAR_POSITION_REL: [0, -5000, 0],
		# 	InputOptions.CAR_VELOCITY_MAG: [0],
		# }
		return self.filter_states(all_state_options)

	def get_output_default(self, outputOption):
		return 1 if outputOption == OutputOptions.THROTTLE else 0

	def map_actions(self, actions):
		action_list = [0 for _ in range(8)]
		action_mapping = {
		  OutputOptions.THROTTLE: 0,
		  OutputOptions.STEER: 1,
		  OutputOptions.PITCH: 2,
		  'yaw': 3,
		  OutputOptions.ROLL: 4,
		  OutputOptions.JUMP: 5,
		  OutputOptions.BOOST: 6,
		  OutputOptions.E_BRAKE: 7,
		}
		for key, index in action_mapping.items():
			action = 0
			if key in self.output_exclude:
				action = self.get_output_default(key)
			elif key == 'yaw' and OutputOptions.STEER not in self.output_exclude:
				action = actions[OutputOptions.STEER][0] - 1
			elif key == OutputOptions.STEER:
				action = actions[key][0] - 1
			else:
				action = actions[key][0]

			action_list[index] = action
		# self.logger.warning(action_list)
		return action_list

	def execute(self, actions):
		action_string = self.match.format_actions(self.map_actions(actions))
		# self.logger.warning(action_string)
		self.pipe_comms.send_message(header=Message.RLGYM_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER, body=action_string)
		message, _ = self.pipe_comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)
		state = self.match.parse_state(message.body)
		next_state = self.gym_to_rl_state(state)


		self.timestep += 1

		if (self.comms is not None):
			serializable_actions = {key:val.tolist() for key, val in actions.items() if key not in self.output_exclude }
			self.comms.outgoing_broadcast.put(serializable_actions)

		self.throttled_log('Ep #{0}--------T:{1}'.format(self.ep, self.timestep))
		reward = self.reward(self.rl_state)
		self.ep_reward += reward
		self.throttled_log("Reward: {0:.2f}".format(self.ep_reward))
		self.executed = True
		# State, terminal, reward
		return self.filter_states(next_state), self.is_terminal(), reward

	def is_terminal(self):
		return 1 if self.timestep >= self.max_timesteps else 0

	def reward(self):
		raise NotImplementedError

	def episode_reward(self, parallel=None):
		return None

	def throttled_log(self, message):
		if self.timestep % self.message_throttle == 0:
			self.logger.warning(message)

	def update_throttle(self, game_tick):
		# kickoff_pause = game_tick.game_info.is_kickoff_pause
		ticks_per_sec = 46.66
		frame_throttle = round(ticks_per_sec / self.frames_per_sec)

		if self.counter == frame_throttle:
			self.counter = 0
			self.executed = False
			return True
		self.counter += 1

		return False

