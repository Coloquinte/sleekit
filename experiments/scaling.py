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


msg = "Data\tMax\tMSE\tDiag"
if args.run_hessian:
    msg += "\tHessian"
if args.run_obq_aware:
    msg += "\tOBQAware"

print(msg)

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, damp=args.damp)
    name = os.path.relpath(root, args.dir)

    gptq_hessian = hessian
    if args.correct_bias:
        eval_hessian = remove_input_bias(hessian, mean)
    else:
        eval_hessian = hessian

    sc = compute_non_saturating_scaling(weight, cb)
    max_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    max_error = quantization_error(weight, max_weight, H=eval_hessian)

    sc = compute_min_mse_scaling(
        weight,
        cb,
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    mse_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    mse_error = quantization_error(weight, mse_weight, H=eval_hessian)

    sc = compute_min_mse_scaling(
        weight,
        cb,
        grid_size=args.grid_size,
        H=np.diag(eval_hessian),
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    diag_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    diag_error = quantization_error(weight, diag_weight, H=eval_hessian)
    msg = f"{name}\t{max_error}\t{mse_error}\t{diag_error}"

    if args.run_hessian:
        sc = compute_min_mse_scaling(
            weight,
            cb,
            grid_size=args.grid_size,
            H=eval_hessian,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
        hessian_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
        hessian_error = quantization_error(weight, hessian_weight, H=eval_hessian)
        msg += f"\t{hessian_error}"

    if args.run_obq_aware:
        sc = compute_min_mse_scaling(
            weight,
            cb,
            grid_size=args.grid_size,
            H=eval_hessian,
            obq=True,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
        obq_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
        obq_error = quantization_error(weight, obq_weight, H=eval_hessian)
        msg += f"\t{obq_error}"

    it.write(msg)
