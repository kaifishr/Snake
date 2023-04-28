"""Trains agent to play Snakes

Uses the defined optimizer procedure to train
the neural network of an agent to play Snakes.

"""
from src.utils import print_args
from src.utils import set_random_seed
from src.argparser import argument_parser
from src.environment import Snakes
from src.model import Model
from src.policy_gradient import PolicyGradient
from src.deep_q_learning import DeepQLearning
from src.trainer import train


if __name__ == "__main__":
    args = argument_parser()
    print_args(args=args)

    env = Snakes(size=args.field_size)

    if args.algorithm == "policy_gradient":
        set_random_seed(seed=args.random_seed)
        model = Model(args)
        agent = PolicyGradient(model=model, args=args)
        train(env=env, agent=agent, args=args)

    elif args.algorithm == "deep_q_learning":
        set_random_seed(seed=args.random_seed)
        model = Model(args)
        agent = DeepQLearning(model=model, args=args)
        train(env=env, agent=agent, args=args)

    else:
        message = f"Reinforcement algorithm {args.algorithm} not implemented"
        raise NotImplementedError(message)
