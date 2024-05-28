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
    "--codebook-size", type=int, default=16, help="Size of the codebook to use"
)
parser.add_argument(
    "--grid-size", type=int, default=100, help="Grid size for error minimization"
)
parser.add_argument(
    "--correct-bias",
    action="store_true",
    help="Use the hessian with bias correction for scaling and evaluation",
)
parser.add_argument(
    "--obq-aware",
    action="store_true",
    help="Run experiments with the slower OBQ-aware scaling",
)
parser.add_argument("--damp", type=float, default=0.01, help="Hessian dampening")
parser.add_argument("--save-figure", type=str, help="Save the figure to this file")
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
rel_error_hessian = []
rel_error_obq = []
rel_error_best = []

if args.obq_aware:
    print("Data\tMax\tMSE\tDiag\tHessian\tOBQAware")
else:
    print("Data\tMax\tMSE\tDiag\tHessian")

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, pcdamp=args.damp)

    gptq_hessian = hessian
    if args.correct_bias:
        eval_hessian = remove_input_bias(hessian, mean)
    else:
        eval_hessian = hessian

    sc = compute_non_saturating_scaling(weight, cb)
    max_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    max_error = quantization_error(weight, max_weight, H=eval_hessian)

    sc = compute_min_mse_scaling(weight, cb, grid_size=args.grid_size)
    mse_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    mse_error = quantization_error(weight, mse_weight, H=eval_hessian)
    best_error = 1.0

    sc = compute_min_mse_scaling(
        weight, cb, grid_size=args.grid_size, H=np.diag(eval_hessian)
    )
    diag_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    diag_error = quantization_error(weight, diag_weight, H=eval_hessian)
    err_diag = diag_error / mse_error
    rel_error_diag.append(err_diag)
    best_error = min(best_error, err_diag)

    sc = compute_min_mse_scaling(weight, cb, grid_size=args.grid_size, H=eval_hessian)
    hessian_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
    hessian_error = quantization_error(weight, hessian_weight, H=eval_hessian)
    err_hessian = hessian_error / mse_error
    rel_error_hessian.append(err_hessian)
    best_error = min(best_error, err_hessian)

    if args.obq_aware:
        sc = compute_min_mse_scaling(
            weight, cb, grid_size=args.grid_size, H=eval_hessian, obq=True
        )
        obq_weight = quantize_with_scaling(weight, sc, cb, H=gptq_hessian)
        obq_error = quantization_error(weight, obq_weight, H=eval_hessian)
        err_obq = obq_error / mse_error
        rel_error_obq.append(err_obq)
        best_error = min(best_error, err_obq)

    name = os.path.relpath(root, args.dir)
    if args.obq_aware:
        it.write(
            f"{name}\t{max_error}\t{mse_error}\t{diag_error}\t{hessian_error}\t{obq_error}"
        )
    else:
        it.write(f"{name}\t{max_error}\t{mse_error}\t{diag_error}\t{hessian_error}")
    rel_error_best.append(best_error)

plt.plot(np.sort(rel_error_diag), label="Diagonal hessian scaling")
plt.plot(np.sort(rel_error_hessian), label="Hessian scaling")
if args.obq_aware:
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
