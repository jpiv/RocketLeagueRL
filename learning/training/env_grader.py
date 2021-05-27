from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

from rlbot.training.training import Grade, Pass, Fail

from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.common_graders.timeout import FailOnTimeout
from rlbottraining.grading.grader import Grader

from tensorforce.environments import Environment

"""
This file shows how to create Graders which specify when the Exercises finish
and whether the bots passed the exercise.
"""

@dataclass
class EnvGrader(Grader):
    env: Environment = None
    runner: any = None
    counter = 0

    def set_match_comms(self, comms):
        self.env and self.env.setComms(comms)

    def on_tick(self, tick):
        gm_tick = tick.game_tick_packet
        
        # self.env.setRLState(gm_tick)
        if self.env and self.env.update_throttle(gm_tick):
            # try:
            #     next(self.runner)
            # except StopIteration:
            #     pass

            if (self.env.fail):
                self.env.fail = False
                return Fail()
            else:
                self.env.done_reseting = True
        