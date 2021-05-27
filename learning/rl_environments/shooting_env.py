from dataclasses import dataclass
from rlbot.utils import public_utils, logging_utils

import sys
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\bot-1\\src')
from util.vec import Vec3
from util.orientation import Orientation, LeftOrientation, relative_location

from .rl_env import RLEnvironment
from .game_values import InputOptions, GameValues
from .env_utils import reward_value

class ShootingEnvironment(RLEnvironment):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.initialize()

	def initialize(self):
		self.goals_scored = 0
		self.new_goal_scored = False

	def reset(self):
		self.new_goal_scored = False
		return super().reset()

	def is_terminal(self):
		return int(self.new_goal_scored)

	def reward(self, rl_state):
		goal_location = Vec3(0, GameValues.GOAL_CENTER_Y, GameValues.GOAL_CENTER_Z)
		ball_velocity = Vec3(*self.game_tick.ball.linear_velocity)	
		ball_position = Vec3(*rl_state[InputOptions.BALL_POSITION])
		goal_direction = relative_location(ball_position, LeftOrientation(), goal_location).normalized()
		ball_velocity_towards_goal = max(ball_velocity.dot(goal_direction), 0)

		# Distance from ball (smallesr)
		ball_dist_value = reward_value(
			rl_state[InputOptions.BALL_DISTANCE][0],
			max_value=GameValues.FIELD_MAX_DISTANCE / 4,
			factor=0.1,
			invert=True,
		)
		# Ball velocity in direction of goal (medium)
		ball_velocity_value = reward_value(
			ball_velocity_towards_goal,
			max_value=GameValues.BALL_MAX_VELOCITY - 2000,
			power=2,
			factor=3,
		)
		# Goal value based on distance with time decay and min (largest)
		base_goal_value = 50
		goals = self.game_tick.blue_score
		new_goal_scored = goals > self.goals_scored 
		goal_value = reward_value(
			# ~x10 at 1sec -> x1 at end of episode
			base_goal_value * (self.max_timesteps / self.timestep),
			min_value=base_goal_value * int(new_goal_scored)
		) if new_goal_scored else 0
		self.goals_scored = goals
		self.new_goal_scored = new_goal_scored

		self.throttled_log("Ball Dist: {0:.2f}".format(ball_dist_value))
		self.throttled_log("Ball Velocity: {0:.2f}".format(ball_velocity_value))
		self.throttled_log("Goal: {0:.2f}".format(goal_value))

		return ball_dist_value + ball_velocity_value + goal_value

