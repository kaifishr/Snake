"""Holds the training method."""
import time

from torch.utils.tensorboard import SummaryWriter

from src.utils import save_config
from src.utils import save_checkpoint
from src.environment import Environment
from src.agent import Agent


def train(env: Environment, agent: Agent, args) -> None:
    """Trains agents with selected reinforcement algorithm."""

    writer = SummaryWriter()
    save_config(writer=writer, args=args)

    for iteration in range(args.num_iterations):
        t0 = time.time()

        events = env.run_episode(agent)

        # Update agent's network.
        agent.step(events)

        # Write metrics for Tensorboard.
        if iteration % args.save_stats_every_n == 0:
            agent.stats["seconds_per_episode"] = time.time() - t0
            for key, value in agent.stats.items():
                if value:
                    writer.add_scalar(f"agent/{key}", value, iteration)

        if iteration % args.save_model_every_n == 0:
            save_checkpoint(model=agent.model, args=args)

    writer.close()
    save_checkpoint(model=agent.model, args=args)
