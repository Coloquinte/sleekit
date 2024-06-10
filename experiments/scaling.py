from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
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
parser.add_argument("--damp", type=float, default=0.01, help="Hessian dampening")
parser.add_argument(
    "--correct-bias",
    action="store_true",
    help="Use the hessian with bias correction for scaling and evaluation",
)
parser.add_argument("--show-figure", action="store_true", help="Show the graph")
parser.add_argument("--save-figure", type=str, help="Save the figure to this file")
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

rel_error_diag = []
rel_error_hessian = []
rel_error_obq = []
rel_error_best = []


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
    remove_dead_values(hessian, weight, pcdamp=args.damp)
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
    best_error = 1.0

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
    err_diag = diag_error / mse_error
    rel_error_diag.append(err_diag)
    best_error = min(best_error, err_diag)
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
        err_hessian = hessian_error / mse_error
        rel_error_hessian.append(err_hessian)
        best_error = min(best_error, err_hessian)
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
        err_obq = obq_error / mse_error
        rel_error_obq.append(err_obq)
        best_error = min(best_error, err_obq)
        msg += f"\t{obq_error}"

    it.write(msg)
    rel_error_best.append(best_error)

if args.save_figure is not None or args.show_figure:
    plt.plot(np.sort(rel_error_diag), label="Diagonal hessian scaling")
    if args.run_hessian:
        plt.plot(np.sort(rel_error_hessian), label="Hessian scaling")
    if args.run_obq_aware:
        plt.plot(np.sort(rel_error_obq), label="OBQ-aware scaling")
    plt.plot(np.sort(rel_error_best), label="Best")
    plt.ylim(bottom=0)
    plt.legend()

    plt.title("Relative error using Hessian scaling")
    plt.xlabel("Layers")
    plt.ylabel("Error relative to MSE")

    if args.save_figure is not None:
        plt.savefig(args.save_figure)
    else:
        plt.show()
