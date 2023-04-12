"""Snakes game engine.

Snakes game with option for arbitrarily many agents.

Typical usage:

    # Play a game against the computer.
    env = Snakes(size=16, num_agents=2)
    env.play()

"""
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.agent import Agent


class Environment:
    """Environment base class."""

    def __init__(self) -> None:
        """Initializes Environment."""
        self.debug = False


class Snakes(Environment):
    """Snakes environment.

    A simple environment for the game Snakes.

    |``````````````````|
    |x -ooo            |
    |     O  oooo      |
    |     oooo  O      |
    |           O      |
    |    oooooooo      |
    |                  |
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
        self.pos_agents = set()  # Keep track of snake's position. TODO: Change later to tuple of sets.

    def _init_agents(self) -> None:
        """Initializes position of agents."""
        self.pos_agents((8, 8))  # TODO: Add random non-overlapping initial positions.

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

        Args:
            index: The index to be converted to a coordinate.

        Returns:
            Tuple with coordinates.
        """
        x, y = divmod(index, self.size)  # index // self.size, index % self.size
        return x, y

    def step(self, action: int, agent_id: int, player: int) -> tuple:
        """Performs a single game move for agent.

        Args:
            action: The action passed by the user or predicted by the neural network.

        Returns:
            A tuple holding state (matrix), reward (scalar), and done (bool) indicating if game is finished.
        """
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
        raise NotImplementedError()

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

    def play(self, model: nn.Module) -> None:
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
                action = int(input(f"Enter an index between [0, 3]: "))
                state, reward, done = self.step(action=action, player=1)

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

    def reset(self) -> torch.Tensor:
        """Resets the playing flied."""
        self.field = torch.zeros(size=(self.size, self.size), dtype=torch.long)
        state = self.field.float()[None, ...]
        self.pos_agents = set()
        return state

    def __repr__(self) -> str:
        """Prints playing field."""
        # prototype = "{:3}" * self.size
        # representation = "\n".join(prototype.format(*row) for row in self.field.tolist())
        # representation = representation.replace("-1", " x").replace("1", "o").replace("0", ".")
        # return "\n" + representation + "\n"
        raise NotImplementedError()
