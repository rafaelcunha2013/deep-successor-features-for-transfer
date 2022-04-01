# -*- coding: UTF-8 -*-
import numpy as np
import random

from tasks.gridworld import Shapes


class MultiagentShapes(Shapes):
    """
    A multiagent discretized version of the gridworld environment introduced in [1]. Here, multiagents learn to
    collect shapes with positive reward, while avoid those with negative reward, and then travel to a fixed goal.
    The gridworld is split into four rooms separated by walls with passage-ways.

    References
    ----------
    [1] Barreto, Andr�, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.
    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new instance of the shape environment that is able to deal with a state with multiple agents and also
        with a joint action.
        """
        super(MultiagentShapes, self).__init__(*args, **kwargs)

    def initialize(self):
        # self.state = (random.choice(self.initial), tuple(0 for _ in range(len(self.shape_ids))))
        self.state = (tuple(self.initial), tuple(0 for _ in range(len(self.shape_ids))))
        my_dict = {}
        key = 0
        for a1 in range(4):
            for a2 in range(4):
                my_dict[key] = (a1, a2)
                key += 1
        self.action_dict = my_dict
        # self.state = ((self.initial[0], self.initial[1]), tuple(0 for _ in range(len(self.shape_ids))))
        return self.state

    def action_count(self):
        return 4  # ** len(self.initial) It was previously increased here

    def agent_count(self):
        return len(self.initial)

    def transition(self, actions):
        agents_position = list(self.state[0])
        # terminal = [False] * len(actions)
        reward = 0.
        # actions = self.action_dict[actions]
        if not isinstance(actions, tuple):
            # actions = list(map(int, str(actions)))
            actions = [actions]
        for i in range(len(actions)):
            (row, col), collected = (self.state[0][i], self.state[1])
            action = actions[i]
            # print(self.state)
            # print(action)
            # perform the movement
            if action == Shapes.LEFT:
                col -= 1
            elif action == Shapes.UP:
                row -= 1
            elif action == Shapes.RIGHT:
                col += 1
            elif action == Shapes.DOWN:
                row += 1
            else:
                raise Exception('bad action {}'.format(action))

            # out of bounds, cannot move
            if col < 0 or col >= self.width or row < 0 or row >= self.height:
                # agent_position[i] = self.state[0][i]
                reward -= 0.1
                continue
                # return self.state, 0., False

            # into a blocked cell, cannot move
            s1 = (row, col)
            if s1 in self.occupied:
                # agent_position[i] = self.state[0][i]
                reward -= 0.1
                continue
                # return self.state, 0., False

            # can now move
            agents_position[i] = s1
            self.state = (tuple(agents_position), collected)
            # self.state = (s1, collected)

            # into a goal cell
            if s1 == self.goal:
                # self.state = (tuple(agents_position), collected)
                return self.state, 1., True

            # into a shape cell
            if s1 in self.shape_ids:
                shape_id = self.shape_ids[s1]
                if collected[shape_id] == 1:

                    # already collected this flag
                    reward -= 0.01
                    continue
                    # return self.state, 0., False
                else:

                    # collect the new flag
                    collected = list(collected)
                    collected[shape_id] = 1
                    collected = tuple(collected)
                    self.state = (tuple(agents_position), collected)
                    # self.state = (s1, collected)
                    reward += self.shape_rewards[self.maze[row, col]]
                    continue
                    # return self.state, reward, False

            # into an empty cell
            # return self.state, 0., False
            reward -= 0.01
        return self.state, reward, False
