"""Snakes game engine.

Snakes game with option for arbitrarily many agents.

Typical usage:

    # Play a game against the computer.
    env = Snakes(size=16, num_agents=2)
    env.play()

"""
import collections
import copy
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.agent import Agent


class Environment:
    """Environment base class."""

    def __init__(self) -> None:
        """Initializes Environment."""
        self.debug = True


class Snakes(Environment):
    """Snakes environment.

    A simple environment for the game Snakes.

    |``````````````````|
    | X  >-ooooooooooo |
    |                O |
    |         oooooooo |
    |         O        |
    |         oooooo   |
    |              O   |
    |           oooo   |
    |..................|

    # Action space

    The action is an integer which can take values [0, 3] indicating
    the direction of the next move:

    .|1|.
    2|x|0
    .|3|.

    # Observation space

    The observation is a PyTorch tensor with shape (size, size) representing the
    playing field.

    # Rewards

        +1 for finding food
        -1 if another agent find the food 
        -1 for hitting a wall
        -1 for hitting the own or others agent's body
        0 if game is draw?

    # Initial state

    An empty playing field represented by a PyTorch tensor with shape (size, size)
    initialized with zeros.

    # Episode end

    The episode ends if any of the following events occur:

    1. The game was lost / won.
    2. Game ends in a draw.
    3. A wrong move was made (forces the agent to learn the game's rules).

    Attributes:
        size: Size of playing field.
        num_agents: Number of competing agents.
        field: PyTorch tensor representing playing field.
    """

    def __init__(self, size: int, num_agents: int = 1) -> None:
        """Initializes a square Tic-tac-toe field."""
        super().__init__()
        self.size = size
        self.num_agents = num_agents
        self.field = torch.zeros(size=(size, size), dtype=torch.long)

        # Keep track of snake's position. TODO: Change later to tuple of sets and deques.
        self.pos_s = None  # TODO: One set should be enough for all agents.
        self.pos_q = None
        self.coord = None  # Holds coordiante tuples of playing field.

        # Lookup table to map actions to moves.
        self.action_to_move = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}

    def _init_agents(self) -> None:
        """Initializes position of agents.
    
        TODO: Add random non-overlapping initial positions.
        """
        pos_init = (2, 2)  # TODO
        self.pos_s.add(pos_init)
        self.pos_q.append(pos_init)
        x, y = pos_init
        self.field[y, x] = 1  # Snake

        food_pos_init = (0, 0)  # TODO
        x, y = food_pos_init
        self.field[y, x] = -1  # Food

        # Create list of all playing field coordinates.
        # Used later to place food.
        l = list(range(self.size))
        self.coord = [(x, y) for x in l for y in l]


    @torch.no_grad()
    def _has_won(self, player: int) -> bool:
        """Checks if player has won the game.

        Args:
            player: The player's id.

        Returns:
            Boolean indicating if game is finished
            (True if game is finished, False otherwise).
        """
        raise NotImplementedError()

    @torch.no_grad()
    def _is_draw(self) -> bool:
        """Checks if the game is tied.

        Return:
            Boolean indicating if game is a draw.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def is_finished(self) -> tuple[bool, int]:
        """Checks if game is finished.

        Returns:
            Tuple with boolean indicating if game
            is finished (True if game is finished, False otherwise)
            and winner of game (player X (1) or player O (-1))
        """
        raise NotImplementedError()

    def is_free(self, x: int, y: int) -> bool:
        """Checks whether field is free or already marked.

        Args:
            x: x-coordinate.
            y: y-coordinate.

        Returns:
            True is field is free. False if field is already marked.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def mark_field(self, x: int, y: int, player: int) -> None:
        """Marks field with either X (1) or O (-1)

        Args:
            x: x-coordinate.
            y: y-coordinate.
            player: Integer representing player A (1) or B (-1).
        """
        raise NotImplementedError()

    def _index_to_coordinate(self, index: int) -> tuple[int, int]:
        """Converts a flat index into a coordinate tuple.

        'x, y = divmod(a, b)' is equivalent to 'x, y = a // b, a % b'

        Args:
            index: The index to be converted to a coordinate.

        Returns:
            Tuple with coordinates.
        """
        x, y = divmod(index, self.size)  # index // self.size, index % self.size
        return x, y

    def run_episode(self, agent_a: Agent, agent_b: Agent) -> tuple:
        """Let agents play one episode of the game.

        The episode stops if the game is won, lost or a draw.

        Args:
            agent_a: Agent holding policy network and reinforcement algorithm.
            agent_b: Agent holding policy network and reinforcement algorithm.

        Returns:
            Tuple for each agent holding states, actions, rewards,
            new_states, and dones of the episode played.
        """
        events_a = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
        events_b = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])

        state = self.reset()
        done = False

        while not done:

            # Agent a
            action = agent_a.get_action(state)
            new_state, reward, done = self.step(action=action, player=-1)

            events_a["states"].append(copy.deepcopy(state))
            events_a["actions"].append(action)
            events_a["rewards"].append(reward)
            events_a["new_states"].append(copy.deepcopy(new_state))
            events_a["dones"].append(done)

            # Player gets negative reward if other player wins.
            if done and reward == 1:
                events_b["rewards"][-1] = -1

            state = new_state

            # Agent b
            if not done:
                action = agent_b.get_action(state)
                new_state, reward, done = self.step(action=action, player=1)

                events_b["states"].append(copy.deepcopy(state))
                events_b["actions"].append(action)
                events_b["rewards"].append(reward)
                events_b["new_states"].append(copy.deepcopy(new_state))
                events_b["dones"].append(done)

                # Player gets negative reward if other player wins.
                if done and reward == 1:
                    events_a["rewards"][-1] = -1

                state = new_state

        return events_a, events_b

    def play(self, model: nn.Module = None) -> None:
        """Runs game in solo mode or against pretrained agents."""

        print("\nGame started.\n")

        done = False
        state = self.reset()

        while not done:

            # print("Machine")
            # action = model.predict(state)
            # state, reward, done = self.step(action=action, player=-1)
            # print(self)
            # if self.debug:
            #     print(f"{state = }")
            #     print(f"{reward = }")
            #     print(f"{done = }")
            # if done:
            #     if reward == 1:
            #         print("You lose.")
            #     elif reward == -1:
            #         print("Illegal move. Computer lost.")
            #     else:
            #         print("Draw.")

            if not done:
                print(self)
                action = int(input(f"Enter an index between [0, 3]: "))
                state, reward, done = self.step(action=action, agent_id=1)

                print(self)

                if self.debug:
                    # print(f"{state = }")
                    print(f"{reward = }")
                    print(f"{done = }")

                if done:
                    if reward == 1:
                        print("You win.")
                    elif reward == -1:
                        print("Illegal move. Computer won.")
                    else:
                        print("Draw.")

    def is_outside(self, x: int, y: int) -> bool:
        """Checks if action leads to collision with wall."""
        if x < 0 or y < 0:
            return True
        elif x == self.size or y == self.size:
            return True
        return False 

    def step(self, action: int, agent_id: int, player: int = 0) -> tuple:
        """Performs a single game move for agent.

        A step consists of moving the snakes head to the new position
        if possible, shorten the snake's tail by unit, and adding new
        food in case no food is left.

        Args:
            action: The action passed by the user or predicted by the 
            neural network.

        Returns:
            A tuple holding state (matrix), reward (scalar), 
            and done (bool) indicating if game is finished.
        """
        # Get head coordinates.
        x, y = self.pos_q[-1]
        x_old, y_old = x, y

        # Compute new head coordinates.
        dx, dy = self.action_to_move[action]
        print(f"{x = }, {y = }")
        print(f"{dx = }, {dy = }")
        x += dx
        y += dy

        # Allows to turn growth on / off.
        allow_growth = True

        # Check for collisions.
        # TODO
        # if self.field[x, y] is is_body(x, y) or is_outside(x, y):
        #    reward = -1.0
        #    done = True
        if self.is_outside(x, y):
            reward = -1.0
            done = True

        else:
            # Check if there is food at the new coordinates.
            if self.field[y, x] == -1:
                # Reward for food found.
                reward = 1.0
                # Snake moves and grows.
                self.pos_q.append((x, y))
                # if not allow_growth:
                #     self.pos_q.popleft()
                # Update playing field.
                x_head, y_head = self.pos_q[-1]
                self.field[y_head, x_head] = 1

                # Add new food.
                print(f"{self.pos_q = }")
                print(f"{self.coord = }")
                # pos_snake = set(self.pos_snake)
                # pos_cells = set(self.pos_cells)
                pos_q = set(self.pos_q)
                coord = set(self.coord)
                empty = coord - pos_q
                print(f"{empty = }")
                self.coord = list(empty)
                if len(self.coord) > 0:
                    idx = random.randrange(len(self.coord))
                    self.coord[idx], self.coord[-1] = self.coord[-1], self.coord[idx]
                    x_food, y_food = self.coord.pop()
                    self.field[y_food, x_food] = -1
                    done = False
                else:
                    # Playing field completely populated by snake(s).
                    done = True

            else:
                # No food found yields no reward.
                reward = 0.0
                done = False

                # Register snake's head.
                self.pos_q.append((x, y))

                # Update playing field.
                x_head, y_head = self.pos_q[-1]
                self.field[y_head, x_head] = 1
                x_tail, y_tail = self.pos_q[0]
                self.field[y_tail, x_tail] = 0

                # Register snake's tail.
                self.pos_q.popleft()

        state = self.field.float()[None, ...]

        return state, reward, done

        # x, y = self._index_to_coordinate(action)
        # if self.is_free(x=x, y=y):
        #     self.mark_field(x=x, y=y, player=player)
        #     is_finished, winner = self.is_finished()
        #     if is_finished and winner == player:
        #         # Maximum reward for winning the game.
        #         reward = 1.0
        #     elif is_finished and winner == 0:
        #         # No reward if game is a draw.
        #         reward = 0.0
        #     else:
        #         # No reward for correctly marking a field.
        #         reward = 0.0
        # else:  # Negative reward and end of game if occupied field is marked.
        #     reward = -1.0
        #     is_finished = True
        # state = self.field.float()[None, ...]
        # done = is_finished
        # return state, reward, done

    def reset(self) -> torch.Tensor:
        """Resets the playing flied."""
        self.field = torch.zeros(size=(self.size, self.size), dtype=torch.long)
        state = self.field.float()[None, ...]
        self.pos_s = set()  
        self.pos_q = collections.deque()
        self._init_agents()
        return state

    def __repr__(self) -> str:
        """Prints playing field."""
        prototype = "{:3}" * self.size
        representation = "\n".join(prototype.format(*row) for row in self.field.tolist())
        representation = representation.replace("-1", " x").replace("1", "o").replace("0", ".")
        return "\n" + representation + "\n"
