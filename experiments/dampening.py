from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import os
import tqdm

import argparse

parser = argparse.ArgumentParser(
    description="Analysis of the effect of dampening on the error"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--codebook-size", type=int, default=4, help="Size of the codebook to use"
)
parser.add_argument(
    "--correct-bias",
    action="store_true",
    help="Use the hessian with bias correction for scaling and evaluation",
)

gp = parser.add_argument_group("Scaling")
gp.add_argument(
    "--scaling",
    type=str,
    choices=["mse", "max", "hessian", "diag", "obq"],
    default="mse",
    help="Scaling method to use",
)
gp.add_argument(
    "--grid-size", type=int, default=100, help="Grid size for error minimization"
)
gp.add_argument(
    "--min-factor",
    type=float,
    default=0.05,
    help="Minimum scaling factor for error minimization",
)
gp.add_argument(
    "--max-factor",
    type=float,
    default=1.0,
    help="Maximum scaling factor for error minimization",
)
args = parser.parse_args()

cb = UniformCodebook(args.codebook_size, -1, 1)

roots = sorted(
    [
        root
        for root, dirs, files in sorted(os.walk(args.dir))
        if "weight.npy" in files and "hessian.npy" in files and "mean.npy" in files
    ]
)

damps = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

msg = "Data\tScaling"
for damp in damps:
    msg += f"\tDamp{damp}"
print(msg)

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, damp=0)
    if args.correct_bias:
        hessian = remove_input_bias(hessian, mean)
    name = os.path.relpath(root, args.dir)

    sc = compute_scaling(
        weight,
        cb,
        H=hessian,
        mode=args.scaling,
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )

    msg = f"{name}\t{args.scaling}"

    for damp in damps:
        dampened_hessian = hessian + damp * hessian.diagonal().mean() * np.eye(
            hessian.shape[0]
        )
        quant_weight = quantize_with_scaling(weight, sc, cb, H=dampened_hessian)
        quant_error = quantization_error(weight, quant_weight, H=hessian)
        msg += f"\t{quant_error}"
    it.write(msg)
