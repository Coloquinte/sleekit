import argparse
import os
import time

import torch
import torch.nn as nn

from sleekit import Sleekit
from datautils import *
from modelutils import *
import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="Model to load; pass `facebook/opt-X` or `bigscience/bloom-X`.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Destination directory for statistics."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling the calibration data (default: %(default)d).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples (default: %(default)d).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.device_count() > 0 else "cpu",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        help="Device to use.",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Force float32 datatype for faster CPU inference",
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Export in numpy format",
    )

    args = parser.parse_args()

    model = get_model(args.model, args.force_fp32)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    extract_statistics(model, dataloader, args)
