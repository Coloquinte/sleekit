from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import os
import tqdm

import argparse


parser = argparse.ArgumentParser(
    description="Analysis of the effect of scaling method on the error"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--codebook-size", type=int, default=4, help="Size of the codebook to use"
)
parser.add_argument("--damp", type=float, default=0.0001, help="Hessian dampening")
parser.add_argument(
    "--correct-bias",
    action="store_true",
    help="Use the hessian with bias correction for scaling and evaluation",
)
gp = parser.add_argument_group("Optimization")
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
gp = parser.add_argument_group("Additional experiments")
gp.add_argument(
    "--run-max",
    action="store_true",
    help="Run experiments with max mode",
)
gp.add_argument(
    "--run-diag",
    action="store_true",
    help="Run experiments with diagonal scaling mode",
)
gp.add_argument(
    "--run-diag1",
    action="store_true",
    help="Run experiments with diag1 scaling mode",
)
gp.add_argument(
    "--run-diag3",
    action="store_true",
    help="Run experiments with diag3 scaling mode",
)
gp.add_argument(
    "--run-diag10",
    action="store_true",
    help="Run experiments with diag10 scaling mode",
)
gp.add_argument(
    "--run-hessian",
    action="store_true",
    help="Run experiments with the slower full hessian scaling",
)
gp.add_argument(
    "--run-obq-aware",
    action="store_true",
    help="Run experiments with the slower OBQ-aware scaling",
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


msg = "Data\tMSE"
modes = ["mse"]
if args.run_max:
    msg += "\tMax"
    modes.append("max")
if args.run_diag:
    msg += "\tDiag"
    modes.append("diag")
if args.run_diag1:
    msg += "\tDiag1"
    modes.append("diag1")
if args.run_diag3:
    msg += "\tDiag3"
    modes.append("diag3")
if args.run_diag10:
    msg += "\tDiag10"
    modes.append("diag10")
if args.run_hessian:
    msg += "\tHessian"
    modes.append("hessian")
if args.run_obq_aware:
    msg += "\tOBQAware"
    modes.append("obq")


print(msg)

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, damp=args.damp)
    if args.correct_bias:
        hessian = remove_input_bias(hessian, mean)
    name = os.path.relpath(root, args.dir)

    msg = f"{name}"
    for mode in modes:
        sc = compute_scaling(
            weight,
            cb,
            H=hessian,
            mode=mode,
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
        quant_weight = quantize_with_scaling(weight, sc, cb, H=hessian)
        quant_error = quantization_error(weight, quant_weight, H=hessian)
        msg += f"\t{quant_error}"
    it.write(msg)
