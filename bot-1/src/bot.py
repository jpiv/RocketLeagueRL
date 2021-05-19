import logging, os
import math

logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
import tensorflow as tf

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

from rlbot.utils import public_utils, logging_utils
from queue import Empty
import os
import sys
sys.path.append('C:\\Users\\John\\Desktop\\stuff\\RLBots\\learning')
from rl_environments.game_values import OutputOptions, InputOptions
from rl_environments.kickoff_env import KickoffEnvironment
from rl_environments.rl_env import RLEnvironment
from tensorforce import Agent, Environment

MODEL = None
# MODEL = 'models/kickoff_no_boost_standalone'

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

        self.last_state = None
        self.agent = None
        self.env = None
        self.ac = None
        self.internals = None

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        if MODEL is not None:
            max_time = 10
            frames_per_sec = 20
            max_timesteps = RLEnvironment.get_max_timesteps(max_time, frames_per_sec)
            self.env = Environment.create(
                environment=KickoffEnvironment,
                max_episode_timesteps=max_timesteps,
                max_time=max_time,
                message_throttle=20,
                frames_per_sec=frames_per_sec,
                input_exclude=[
                    InputOptions.BALL_POSITION_REL,
                    InputOptions.BALL_DIRECTION,
                    InputOptions.CAR_POSITION_REL,
                    InputOptions.CAR_VELOCITY_MAG,
                ],
                output_exclude=[
                    OutputOptions.BOOST,
                    OutputOptions.STEER,
                    OutputOptions.E_BRAKE,
                    OutputOptions.THROTTLE,
                    OutputOptions.ROLL,
                ]
            )

            directory='../learning/training/{0}'.format(MODEL)
            filename='agent'
            agent = os.path.join(directory, os.path.splitext(filename)[0] + '.json') 

            if not os.path.isfile(agent):
                logging_utils.log_warn(os.getcwd(), {})
                raise Exception('Model file doesn\'t exist')
            
            self.agent = Agent.load(
                directory=os.path.abspath(directory),
                environment=self.env,
                format='checkpoint',
            )
            self.env.reset()

    def render(self, packet):
        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location

        if car_location.dist(ball_location) > 1500:
            # We're far away from the ball, let's try to lead it a little bit
            ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
            ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 2)

            # ball_in_future might be None if we don't have an adequate ball prediction right now, like during
            # replays, so check it to avoid errors.
            if ball_in_future is not None:
                target_location = Vec3(ball_in_future.physics.location)
                self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        self.render(packet)

        my_car = packet.game_cars[self.index]
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        target_location = ball_location

        if car_location.dist(ball_location) > 1500:
            ball_prediction = self.get_ball_prediction_struct()
            ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 2)

            if ball_in_future is not None:
                target_location = Vec3(ball_in_future.physics.location)

        # Drive at ball
        controls = self.last_state or SimpleControllerState(throttle=1)
        # controls.steer = steer_toward_target(my_car, target_location)

        self.set_controls_from_model(packet, controls)

        self.last_state = controls

        return controls

    def set_controls_from_model(self, tick, controls):
        actions = self.get_actions(tick)

        if (len(actions) > 1):
            controls.throttle = actions[OutputOptions.THROTTLE][0] - 1 if OutputOptions.THROTTLE in actions else controls.throttle
            controls.pitch = actions[OutputOptions.PITCH][0] - 1 if OutputOptions.PITCH in actions else controls.pitch
            controls.roll = actions[OutputOptions.ROLL][0] - 1 if OutputOptions.ROLL in actions else controls.roll
            controls.yaw = actions[OutputOptions.STEER][0] - 1 if OutputOptions.STEER in actions else controls.yaw
            controls.steer = actions[OutputOptions.STEER][0] - 1 if OutputOptions.STEER in actions else controls.steer
            controls.boost = actions[OutputOptions.BOOST][0] if OutputOptions.BOOST in actions else controls.boost
            controls.jump = actions[OutputOptions.JUMP][0] if OutputOptions.JUMP in actions else controls.jump
            controls.handbrake = actions[OutputOptions.E_BRAKE][0] if OutputOptions.E_BRAKE in actions else controls.handbrake

        return controls

    def get_actions(self, tick):
        actions = {}
        if self.env is not None and self.env.update_throttle(tick):
            states = self.env.setRLState(tick)
            actions = self.agent.act(states)
            states, terminal, reward = self.env.execute(actions=actions)
            self.agent.observe(terminal=terminal, reward=reward)
            if terminal: self.env.reset()
        else:
            try:
                actions = self.matchcomms.incoming_broadcast.get(block=False)
            except Empty:
                pass
        return actions


