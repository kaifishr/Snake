"""Argument parser

Holds environment and learning parameters.
"""
import argparse


def argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="Snake",
        description="Snake with reinforcement learning.",
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

    # parser.add_argument("--num-agents", dest="num_agents", default=1, type=int)

    ###############
    # Environment #
    ###############
    parser.add_argument(
        "--field-size", 
        dest="field_size", 
        default=8, 
        type=int
    )

    parser.add_argument(
        "--num-frames", 
        dest="num_frames", 
        default=3, 
        type=int,
        help="Number of past frames that are fed into the policy network.",
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
        type=bool
    )

    parser.add_argument(
        "--forbid-growth",
        action="store_true",
        dest="forbid_growth",
    )

    # parser.add_argument(
    #     "--encode-length",
    #     dest="encode_length",
    #     default=True,
    #     type=bool
    # )

    ###########
    # Trainer #
    ###########
    parser.add_argument(
        "--learning-rate", 
        dest="learning_rate", 
        default=2e-4, 
        type=float
    )

    parser.add_argument(
        "--num-iterations", 
        dest="num_iterations", 
        default=1_000_000, 
        type=int
    )

    parser.add_argument(
        "--num-episodes", 
        dest="num_episodes", 
        default=8,
        type=int,
        help="Number of episodes per optimization step. (Deep Q-Learning)",
    )

    parser.add_argument(
        "--gamma",
        dest="gamma",
        help="Discount or forgetting factor. 0 <= gamma <= 1.",
        default=0.9,
        type=float,
    )

    parser.add_argument(
        "--epsilon",
        dest="epsilon",
        help="Initial epsilon-greedy value (exploration rate).",
        default=0.5,
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
        default=0.999995,
        type=float,
    )

    parser.add_argument(
        "--memory-size",
        dest="memory_size",
        help="Replay memory size. Set to 1 for no memory.",
        default=5_000,
        type=int,
    )

    parser.add_argument(
        "--batch_size", 
        dest="batch_size", 
        default=256,
        type=int
    )

    #############################
    # Model / policy parameters #
    #############################
    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="Defines which model to load.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--dropout-rate",
        dest="dropout_rate",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--num-layers", 
        dest="num_layers", 
        default=4, 
        type=int
    )
    
    parser.add_argument(
        "--num-channels", 
        dest="num_channels", 
        default=32, 
        type=int
    )

    ###############
    # Tensorboard #
    ###############
    parser.add_argument(
        "--save-stats-every-n",
        dest="save_stats_every_n",
        help="Defines how often statistics are saved.",
        default=100,
        type=int,
    )

    ###############
    # Checkpoints #
    ###############
    parser.add_argument(
        "--save-model-every-n",
        dest="save_model_every_n",
        help="Defines how often the model is saved.",
        default=5000,
        type=int,
    )

    return parser.parse_args()
