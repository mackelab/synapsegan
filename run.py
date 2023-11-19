import argparse
import wandb
from os.path import join
from os import makedirs
import importlib


def main(args):
    """Set up pipeline for training."""
    # Imports
    task = importlib.import_module("tasks.%s" % args.task_name)
    factory = task.Factory(resume=args.resume, resume_dir=args.resume_dir)

    # Make logger and log directory
    makedirs(join("..", "runs", "%s" % args.task_name), exist_ok=True)
    wandb.init(project=args.project_name,
               group=args.group_name,
               id=args.run_id,
               resume=args.resume,
               config=factory.fconf,
               dir=join("..", "runs", "%s" % args.task_name)
               )
    run = wandb.run

    with run:
        optimizer = factory.make_optimizer(logger=run)
        print("Train")
        optimizer.train(args.epochs, 100)
        factory.post_processing(optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--project_name", type=str, default="SynapseGAN")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    main(parser.parse_args())
