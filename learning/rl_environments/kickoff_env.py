from dataclasses import dataclass
from rlbot.utils import public_utils, logging_utils

from .rl_env import RLEnvironment
from .game_values import InputOptions

class KickoffEnvironment(RLEnvironment):
	def reward(self, rl_state):
		# Actually 6000
		MAX_BALL_VEL = 3000
		MAX_CAR_VELOCITY = 2300
		MAX_DRIVING_VELOCITY = 1410
		BASE_BALL_VEL =  MAX_DRIVING_VELOCITY - 200
		normalized_ball_velocity = (
			pow(max(abs(self.ball_velocity), 0), 4) * 10
			/ pow(MAX_BALL_VEL, 4)
		)

		car_velocity_abv_baseline = max(sum(abs(v) for v in self.rl_state['car_velocity'][:-1]) - MAX_DRIVING_VELOCITY, 0)
		normalized_car_velocity = car_velocity_abv_baseline / (MAX_CAR_VELOCITY - MAX_DRIVING_VELOCITY)
		normalized_ball_dist = max(
			0.1 * (1 - pow(self.rl_state[InputOptions.BALL_DISTANCE][0], 2) / pow(3000, 2)), 0
		)

		self.throttled_log("Car: {0:.2f}".format(normalized_car_velocity))
		self.throttled_log("Ball: {0:.2f}".format(normalized_ball_velocity))
		self.throttled_log("Dist: {0:.2f}".format(normalized_ball_dist))

		return normalized_ball_velocity

