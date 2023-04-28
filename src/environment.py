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
        """Initializes a square playing field."""
        super().__init__()
        self.size = size
        self.num_agents = num_agents
        self.field = torch.zeros(size=(size, size))

        # Keep track of snake's position. 
        # TODO: Change later to tuple of sets and deques.
        self.pos_s = None  # TODO: One set should be enough for all agents.
        self.pos_q = None
        self.coord = None  # Holds coordiante tuples of playing field.

        # Lookup table to map actions to moves.
        self.action_to_move = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}

        # Rendering
        self.is_render = True
        if self.is_render:
            self._init_render()

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
    def is_finished(self) -> tuple[bool, int]:
        """Checks if game is finished.

        Returns:
            Tuple with boolean indicating if game
            is finished (True if game is finished, False otherwise)
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

    def run_episode(self, agent: Agent) -> tuple:
        """Let agents play one episode of the game.

        The episode stops if the game is won, lost or a draw.

        Args:
            agent: Agent holding policy network and reinforcement algorithm.

        Returns:
            Tuple for each agent holding states, actions, rewards,
            new_states, and dones of the episode played.
        """
        events = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])

        state = self.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            new_state, reward, done = self.step(action=action)

            events["states"].append(copy.deepcopy(state))
            events["actions"].append(action)
            events["rewards"].append(reward)
            events["new_states"].append(copy.deepcopy(new_state))
            events["dones"].append(done)

            state = new_state

        return events 

    def is_outside(self, x: int, y: int) -> bool:
        """Checks for collision with wall.

        Args:
            x: x-coordinate.
            y: y-coordinate.

        Returns:
            True if snake attempts to move outside of playing field. False otherwise.
        """
        if x == -1 or y == -1:
            return True
        elif x == self.size or y == self.size:
            return True
        return False

    def is_overlap(self, x: int, y: int) -> bool:
        """Checks for collision with body.

        Args:
            x: x-coordinate.
            y: y-coordinate.

        Returns:
            True if snake collides with body. False otherwise.
        """
        if (x, y) in set(self.pos_q):
            # if (x, y) == self.pos_q[0]:  # Allow 'collision' with tail.
            #     return False
            return True
        return False

    def step(self, action: int) -> tuple:
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
        print("\n **********\n", "** STEP **", "\n **********\n")
        # Get head coordinates.
        x, y = self.pos_q[-1]

        # Compute new head coordinates.
        dx, dy = self.action_to_move[action]
        x += dx
        y += dy

        # Allows to turn growth on / off.
        # allow_growth = True

        # Check for collisions.
        if self.is_outside(x, y) or self.is_overlap(x, y):
            reward = -1.0
            done = True
        else:
            # Check if there is food at the new coordinates.
            if self.field[y, x] == -1:
                # Reward for food found.
                reward = 1.0
                # Snake moves and grows.
                self.pos_q.append((x, y))
                # Update playing field.
                # x_head, y_head = self.pos_q[-1]
                # self.field[y_head, x_head] = 1

                # Add new food.
                pos_q = set(self.pos_q)
                coord = set(self.coord)
                empty = list(coord - pos_q)
                if len(empty) > 0:
                    x_food, y_food = random.choice(empty)
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
                x_tail, y_tail = self.pos_q[0]
                self.field[y_tail, x_tail] = 0
                # x_head, y_head = self.pos_q[-1]
                # self.field[y_head, x_head] = 1

                # Register snake's tail.
                self.pos_q.popleft()

            # Encodes snake's body in playing field.
            encoding = torch.linspace(start=1.0, end=2.0, steps=len(self.pos_q))
            for i, (x, y) in enumerate(self.pos_q):
                self.field[y, x] = encoding[i]

        state = self.field.float()[None, ...]

        return state, reward, done

    def play(self, model: nn.Module = None) -> None:
        """Runs game in solo mode."""

        print("\nGame started.\n")
        print(f"Enter an index between [0, 3]. Pres 'q' to quit.")

        done = False
        state = self.reset()
        print(self)
        if self.is_render:
            self._render()

        while not done:

            command = input("Enter command: ")

            if command == "q":
                exit("Game quit.")
            elif command.isnumeric():
                action = int(command)
                is_valid_command = 0 <= action <= 3

            if is_valid_command:

                state, reward, done = self.step(action=action)
                print(self)

                if self.debug:
                    print(f"{state = }")
                    print(f"{reward = }")
                    print(f"{done = }")

                if done:
                    if reward == 1:
                        print("You win.")
                    elif reward == -1:
                        print("Illegal move. Computer won.")
                    else:
                        print("Draw.")
            else:
                print("Invalid input. Enter an integer from 0 to 3.")

            if self.is_render:
                self._render()

    def _init_render(self) -> None:
        self.fig, axis = plt.subplots()
        self.fig.canvas.manager.set_window_title("Snakes")
        field = self.field.numpy()
        extent = (0, field.shape[1], field.shape[0], 0)
        self.img = axis.imshow(field, vmin=-1.0, vmax=2.0, cmap="bwr", 
                               interpolation='none', aspect='equal', 
                               extent=extent)
        axis.grid(color="k", linewidth=1)
        self.fig.canvas.draw()
        plt.show(block=False)

    def _render(self) -> None:
        """Renders playing field.

        Consider this answer for faster rendering:
        https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
        or use PyGame.
        """
        field = self.field.numpy()
        self.img.set_data(field)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self) -> torch.Tensor:
        """Resets the playing flied."""
        self.field = torch.zeros(size=(self.size, self.size))
        state = self.field[None, ...]
        self.pos_s = set()
        self.pos_q = collections.deque()
        self._init_agents()
        return state

    def __repr__(self) -> str:
        """Prints playing field."""
        field = torch.where(self.field >= 1.0, 1.0, self.field).long()
        prototype = "{:3}" * self.size
        representation = [prototype.format(*row) for row in field.tolist()]
        representation = "\n".join(representation)
        substitutes = [("-1", " x"), ("1", "o"), ("0", ".")]
        for item1, item2 in substitutes:
            representation = representation.replace(item1, item2)
        return "\n" + representation + "\n"
