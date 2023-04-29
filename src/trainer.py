"""Holds the training method."""
import time

from torch.utils.tensorboard import SummaryWriter

from src.utils import save_checkpoint
from src.environment import Environment
from src.agent import Agent


def train(env: Environment, agent: Agent, args) -> None:
    """Trains agents with selected reinforcement algorithm."""

    writer = SummaryWriter()

    for episode in range(args.num_episodes):

        t0 = time.time()

        # Run episode and let the agents compete.
        events = env.run_episode(agent)

        # Update network.
        agent.step(events)

        if episode % 50 == 0:  # TODO: Move magic number to arguments.
            for key, value in agent.stats.items():
                if value:
                    writer.add_scalar(f"agent/{key}", value, episode)
            print(f"{time.time() - t0:.2}")

    writer.close()

    save_checkpoint(model=agent.model, args=args)
