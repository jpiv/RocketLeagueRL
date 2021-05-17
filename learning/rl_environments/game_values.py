from enum import Enum

class InputOptions(str, Enum):
	BALL_DISTANCE = 'dist'
	BALL_POSITION = 'ball_pos'
	CAR_ORIENTATION = 'car_orientation'
	CAR_HEIGHT = 'car_height'
	CAR_POSITION = 'car_position'
	CAR_VELOCITY = 'car_velocity'
	JUMPED = 'jumped'
	DOUBLE_JUMPED = 'double_jumped'

class OutputOptions(str, Enum):
	JUMP = 'do_jump'
	PITCH = 'stick_magnitude'
