"""Play against an agent."""
from src.utils import print_args
from src.utils import load_checkpoint
from src.argparser import argument_parser
from src.model import Model
from src.environment import Snakes


if __name__ == "__main__":
    args = argument_parser()
    print_args(args=args)

    if args.play_mode == "agent":
        model = Model(args=args)
        load_checkpoint(model=model, args=args)
        env = Snakes(args=args)
        env.play_agent(model=model)
    else:  # solo play mode
        env = Snakes(args=args)
        env.play()
