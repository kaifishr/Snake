"""Argument parser.

Holds environment and learning parameters.

"""
import argparse


def argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="Snakes",
        description="Trains agent to play Snakes with reinforcement learning.",
    )

    parser.add_argument(
        "--random-seed", 
        dest="random_seed", 
        default=42, 
        type=int
    )

    #########
    # Agent #
    #########

    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        help="Reinforcement learning algorithm.",
        default="policy_gradient",
        choices=["policy_gradient", "deep_q_learning"],
        type=str,
    )

    parser.add_argument(
        "--num-agents", 
        dest="num_agents", 
        default=1, 
        type=int
    )

    ###############
    # Environment #
    ###############

    parser.add_argument(
        "--field-size", 
        dest="field_size", 
        default=3, 
        type=int
    )

    parser.add_argument(
        "--mode",
        dest="mode",
        default="human",
        choices=["human", "agent"],
        type=str,
    )

    parser.add_argument(
        "--render",
        dest="render",
        default=False,
        type=str
    )

    ###########
    # Trainer #
    ###########

    parser.add_argument(
        "--learning-rate", 
        dest="learning_rate", 
        default=1e-4, 
        type=float
    )

    parser.add_argument(
        "--num-episodes", 
        dest="num_episodes", 
        default=1_000_000, 
        type=int
    )

    parser.add_argument(
        "--gamma",
        dest="gamma",
        help="Discount or forgetting factor. 0 <= gamma <= 1.",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--epsilon",
        dest="epsilon",
        help="Epsilon-greedy value (exploration rate).",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--epsilon-min",
        dest="epsilon_min",
        help="Minimum epsilon-greedy value.",
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "--decay-rate",
        dest="decay_rate",
        help="Decay rate for epsilon-greedy value.",
        default=0.99999,
        type=float,
    )

    parser.add_argument(
        "--memory-size",
        dest="memory_size",
        help="Replay memory size. Set to 1 for no memory.",
        default=10_000,
        type=int,
    )

    parser.add_argument(
        "--batch_size", 
        dest="batch_size", 
        default=128,
        type=int
    )

    #############################
    # Model / policy parameters #
    #############################

    parser.add_argument(
        "--dropout-probability",
        dest="dropout_probability",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--num-layers", 
        dest="num_layers", 
        default=1,
        type=int
    )

    parser.add_argument(
        "--hidden-units", 
        dest="num_hidden_units", 
        default=128, 
        type=int
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="Defines which model to load.",
        default=None,
        type=str,
    )

    #######################
    # Tensorboard / Model #
    #######################
    parser.add_argument(
        "--save-stats-every-n",
        dest="save_stats_every_n",
        help="Defines how often statistics are saved.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--save-model-every-n",
        dest="save_model_every_n",
        help="Defines how often the model is saved.",
        default=1000,
        type=int,
    )
    # TODO Arguments
    # - allow snakes to grow on / off
    # - colorcode snakes on / off

    return parser.parse_args()
