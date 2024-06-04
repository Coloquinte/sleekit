from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import os
import tqdm

import argparse

parser = argparse.ArgumentParser(
    description="Analysis of the effect of GPTQ ordering on the error"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--codebook-size", type=int, default=4, help="Size of the codebook to use"
)
parser.add_argument("--damp", type=float, default=0.01, help="Hessian dampening")
parser.add_argument(
    "--correct-bias",
    action="store_true",
    help="Use the hessian with bias correction for evaluation",
)
parser.add_argument("--show-figure", action="store_true", help="Show the graph")
parser.add_argument("--save-figure", type=str, help="Save the figure to this file")

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

cb = Codebook.uniform(args.codebook_size, -1, 1)

roots = sorted(
    [
        root
        for root, dirs, files in sorted(os.walk(args.dir))
        if "weight.npy" in files and "hessian.npy" in files and "mean.npy" in files
    ]
)

rel_error_diag = []
rel_error_err = []
rel_error_sqerr = []

print("Data\tScaling\tDiag\tDiagErr\tDiagSqErr")
it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, pcdamp=args.damp)
    if args.correct_bias:
        hessian = remove_input_bias(hessian, mean)

    if args.scaling == "mse":
        sc = compute_min_mse_scaling(
            weight,
            cb,
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
    elif args.scaling == "max":
        sc = compute_non_saturating_scaling(weight, cb)
    elif args.scaling == "hessian":
        sc = compute_min_mse_scaling(
            weight,
            cb,
            H=hessian,
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
    elif args.scaling == "diag":
        sc = compute_min_mse_scaling(
            weight,
            cb,
            H=np.diag(hessian),
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
    elif args.scaling == "obq":
        sc = compute_min_mse_scaling(
            weight,
            cb,
            H=hessian,
            obq=True,
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
    else:
        raise RuntimeError(f"Unknown scaling {args.scaling}")

    diag_weight = quantize_with_scaling(weight, sc, cb, H=hessian, act_order="diag")
    diag_error = quantization_error(weight, diag_weight, H=hessian)
    err_weight = quantize_with_scaling(
        weight, sc, cb, H=hessian, act_order="err"
    )
    err_error = quantization_error(weight, err_weight, H=hessian)
    sqerr_weight = quantize_with_scaling(
        weight, sc, cb, H=hessian, act_order="sqerr"
    )
    sqerr_error = quantization_error(weight, sqerr_weight, H=hessian)
    name = os.path.relpath(root, args.dir)
    it.write(
        f"{name}\t{args.scaling}\t{diag_error}\t{err_error}\t{sqerr_error}"
    )

    rel_error_diag.append(diag_error / diag_error)
    rel_error_err.append(err_error / diag_error)
    rel_error_sqerr.append(sqerr_error / diag_error)

if args.save_figure is not None or args.show_figure:
    plt.plot(np.sort(rel_error_diag), label="Diagonal ordering")
    plt.plot(np.sort(rel_error_err), label="Diag * error ordering")
    plt.plot(np.sort(rel_error_sqerr), label="Diag * squared error ordering")
    plt.ylim(bottom=0)
    plt.legend()

    plt.title("Relative error with different GPTQ orderings")
    plt.xlabel("Layers")
    plt.ylabel("Error relative to diagonal ordering")

    if args.save_figure is not None:
        plt.savefig(args.save_figure)
    else:
        plt.show()
