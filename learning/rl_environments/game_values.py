from enum import Enum
import sys
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\bot-1\\src')
from util.vec import Vec3

class InputOptions(str, Enum):
	BALL_DISTANCE = 'dist'
	BALL_POSITION = 'ball_pos'
	# Ball position relative to car
	BALL_POSITION_REL = 'ball_position_relative'
	# Normalized ball position relative to car
	BALL_DIRECTION = 'ball_direction'
	CAR_ORIENTATION = 'car_orientation'
	CAR_HEIGHT = 'car_height'
	CAR_POSITION = 'car_position'
	# Car position relative to goal
	CAR_POSITION_REL = 'car_position_rel'
	CAR_VELOCITY = 'car_velocity'
	CAR_VELOCITY_MAG = 'car_velocity_mag'
	JUMPED = 'jumped'
	DOUBLE_JUMPED = 'double_jumped'

class OutputOptions(str, Enum):
	JUMP = 'do_jump'
	PITCH = 'stick_magnitude'
	ROLL = 'roll'
	STEER = 'steer'
	BOOST = 'boost'
	THROTTLE = 'throttle'
	E_BRAKE = 'e_brake'

class GameValues(float, Enum):
	GOAL_CENTER_Y = 5120
	GOAL_CENTER_Z = 321.39
	# Max field Y value + goal depth
	FIELD_MAX_Y = 5120 + 881
	FIELD_MIN_Y = -5120 - 881
	# Diangonal  from corner to corner of field
	FIELD_MAX_DISTANCE = 13272
	FIELD_HEIGHT = 2045
	CAR_MAX_VELOCITY = 2301
	CAR_DRIVING_VELOCITY = 1410
	BALL_MAX_VELOCITY = 6001

