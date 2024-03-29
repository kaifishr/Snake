"""Snake game engine.

Snake game with option for arbitrarily many agents.

Typical usage:

    # Play a game against the computer.
    env = Snake(size=16, num_agents=2)
    env.play()

"""
import collections
import copy
import random
import time

import matplotlib.pyplot as plt
import torch

from src.agent import Agent


class Environment:
    """Environment base class."""

    def __init__(self) -> None:
        """Initializes Environment."""
        self.debug = True


class Snake(Environment):
    """Snake environment.

    A simple environment for the game Snake.

    |``````````````````|
    | X  >oooooooooooo |
    |                O |
    |         oooooooo |
    |         O        |
    |     O   oooooo   |
    |     O        O   |
    |     oooooooooo   |
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
        -1 for exceeding maximum number of steps
        0 if game is draw?

    # Initial state

    An empty playing field represented by a PyTorch tensor with shape (size, size)
    initialized with zeros.

    # Episode end

    The episode ends if any of the following events occur:

    1. The game was lost / won.
    2. Game ends in a draw.
    3. A wrong move was made (forces the agent to learn the game's rules).
    4. Maximum number of steps has been exceeded.

    Attributes:
        size: Size of playing field.
        num_agents: Number of competing agents.
        field: PyTorch tensor representing playing field.
        pos_q:
        coord:
        action_to_move:
        key_to_action:
        max_steps_episode:
        step_counter:
    """

    def __init__(self, args) -> None:
        """Initializes a square playing field.
        
        Args:
            args: Command line arguments.
        """
        super().__init__()

        self.forbid_growth = args.forbid_growth
        self.size = args.field_size
        self.num_episodes = args.num_episodes
        self.num_frames = args.num_frames
        self.encode_length = args.encode_length
        self.field = torch.zeros(size=(self.size, self.size)) # TODO: Required?

        # Keep track of snake's position.
        self.pos_q = None
        self.coord = None  # Holds coordinate tuples of playing field.

        # Lookup table to map actions to moves (left, right, up, down).
        self.action_to_move = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}
        # TODO: Move this to a render class?
        self.key_to_action = {"w": 1, "a": 2, "s": 3, "d": 0}

        # Maximum number of steps an agent can take to get to the next treat.
        self.max_steps_episode = self.size * self.size
        self.step_counter = 0

        # Initialize past frames with tensor filled with scalar value 0.
        self.frames = None
        self.frames = self._init_frames()  # TODO: Required?

    def _init_frames(self) -> None:
        """Initialize past frames with tensor filled with scalar value 0."""
        frames = collections.deque(maxlen=self.num_frames)
        for _ in range(self.num_frames):
            frames.append(torch.zeros(size=(self.size, self.size)))
        return frames

    def _init_agent(self) -> None:
        """Initializes position of agent."""
        # Create list of all playing field coordinates.
        # Used later to place food.
        l = list(range(self.size))
        self.coord = [(x, y) for x in l for y in l]

        # Set initial position of snake.
        pos_init = random.choice(self.coord)
        self.pos_q.append(pos_init)
        x, y = pos_init
        self.field[y, x] = 1.0

        self._add_food()

    def _add_food(self) -> None:
        """Adds food at random empty location in playing field."""
        pos_q = set(self.pos_q)
        coord = set(self.coord)
        empty = list(coord - pos_q)

        if len(empty) > 0:
            x_food, y_food = random.choice(empty)
            self.field[y_food, x_food] = -1
            done = False
        else:
            # Playing field completely populated by the snake.
            done = True

        return done

    def run_episode(self, agent: Agent) -> tuple:
        """Let agents play `n` episodes of the game.

        The episode stops if the game is won, lost or a draw.

        Args:
            agent: Agent holding policy network and reinforcement algorithm.

        Returns:
            Tuple for each agent holding states, actions, rewards,
            new_states, and dones of the episode played.
        """
        events = {
            "states": [],
            "actions": [],
            "rewards": [],
            "new_states": [],
            "dones": [],
        }
        for _ in range(self.num_episodes):
            # TODO: 'state' is reference to 'self.field'.
            # TODO: Check if same bug is also in TicTacToe project.
            state = self._reset()
            done = False
            while not done:
                events["states"].append(copy.deepcopy(state))
                action = agent.get_action(state)
                new_state, reward, done = self.step(action=action)
                events["actions"].append(action)
                events["rewards"].append(reward)
                events["dones"].append(done)
                self.frames.append(copy.deepcopy(new_state))
                state = torch.stack(list(self.frames), dim=0)
                events["new_states"].append(copy.deepcopy(state))

        return events

    def _is_outside(self, x: int, y: int) -> bool:
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

    def _is_overlap(self, x: int, y: int) -> bool:
        """Checks for collision with body.

        Args:
            x: x-coordinate.
            y: y-coordinate.

        Returns:
            True if snake collides with body. False otherwise.
        """
        if (x, y) in set(self.pos_q):  # TODO: Check this.
            # if (x, y) == self.pos_q[0]:  # Allow 'collision' with tail.
            #     return False
            return True
        return False

    def step(self, action: int) -> tuple:
        """Performs a single game move for agent.

        A step consists of moving the snake's head to the new position
        if possible, shorten the snake's tail by unit, and adding new
        food in case no food is left.

        Args:
            action: The action passed by the user or predicted by the
            neural network.

        Returns:
            A tuple holding state (matrix), reward (scalar),
            and done (bool) indicating if game is finished.
        """
        # Count number of steps since last food encounter.
        self.step_counter += 1

        # Get head coordinates.
        x, y = self.pos_q[-1]

        # Compute new head coordinates.
        dx, dy = self.action_to_move[action]
        x += dx
        y += dy

        if self.step_counter == self.max_steps_episode:
            # print(f"Maximum steps exceeded ({self.max_steps_episode}).")
            reward = -1.0
            done = True
        elif self._is_outside(x, y) or self._is_overlap(x, y):
            # Negative reward and end of game in case of collisions with body, 
            # other agents or if snake hits the playing field boundary.
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
                if self.forbid_growth:
                    x_tail, y_tail = self.pos_q[0]
                    self.field[y_tail, x_tail] = 0
                    self.pos_q.popleft()

                # Add new food.
                done = self._add_food()

                # Set counter back.
                self.step_counter = 0
            else:
                # No food found yields negative reward to encourage agent
                # not to dawdling too much.
                reward = - 1.0 / (self.size * self.size)
                done = False

                # Register snake's head.
                self.pos_q.append((x, y))

                # Update playing field.
                x_tail, y_tail = self.pos_q[0]
                self.field[y_tail, x_tail] = 0

                # Register snake's tail.
                self.pos_q.popleft()

            # Encode snake in playing field.
            # encoding = torch.linspace(start=1.0, end=2.0, steps=len(self.pos_q))
            for i, (x, y) in enumerate(self.pos_q):
                self.field[y, x] = 1.0
                if self.encode_length:
                    self.field[y, x] += i / len(self.pos_q)

        state = self.field

        return state, reward, done

    def play(self) -> None:
        """Runs game in solo mode."""

        print("\nGame started.\n")
        print(f"Use 'WASD' keys to move. Press 'q' to quit.")

        self._init_render()

        done = False
        # state = self._reset()
        print(self)

        self._render()

        while not done:
            key = input("Enter command: ")

            if key == "q":
                exit("Game quit.")

            elif key in self.key_to_action:
                action = self.key_to_action[key]
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
                print(f"Invalid input. Use 'WASD' keys to move.")

            self._render()

    def play_agent(self, model: torch.nn.Module) -> None:
        """Runs game with model."""

        sleep_seconds = 0.05
        done = False
        step = 0

        self._init_render()
        state = self._reset()
        self._render(step=step)


        while not done:

            step += 1
            time.sleep(sleep_seconds) 

            action = model.predict(state)
            new_state, reward, done = self.step(action=action)
            self.frames.append(copy.deepcopy(new_state))
            state = torch.stack(list(self.frames), dim=0)

            if self.debug:
                print(f"{action = }")
                print(f"{state = }")
                print(f"{reward = }")
                print(f"{done = }\n")

            if done and reward == -1:
                print("Game over.")

            self._render(step=step)

    def _init_render(self) -> None:
        self.fig, axis = plt.subplots()
        self.fig.canvas.manager.set_window_title("Snake")
        field = self.field.numpy()
        extent = (0, field.shape[1], field.shape[0], 0)
        self.img = axis.imshow(
            field,
            vmin=-1.0,
            vmax=1.0,
            cmap="bwr",
            interpolation="none",
            aspect="equal",
            extent=extent,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        self.fig.canvas.draw()
        plt.show(block=False)

    def _render(self, step: int = 0) -> None:
        """Renders playing field.

        Args:
            step: Current step taken by agent.

        Consider this answer for faster rendering:
        https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
        or use PyGame.
        """
        field = self.field.numpy()
        self.img.set_data(field)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.savefig(f"out/{step}", bbox_inches="tight")

    def _reset(self) -> torch.Tensor:
        """Resets the environment.
        
        Returns:
            Tensor holding state.
        """
        # Reset playing field.
        self.field = torch.zeros(size=(self.size, self.size))
        # Reset snake.
        self.pos_q = collections.deque()
        # Initialize new agents.
        self._init_agent()

        self.step_counter = 0

        frames = self._init_frames()
        state = torch.stack(tensors=list(frames), dim=0)

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