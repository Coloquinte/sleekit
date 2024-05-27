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
rel_error_best = []

print("Data\tMax\tMSE\tDiag\tHessian")
it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, pcdamp=args.damp)

    sc = compute_non_saturating_scaling(weight, cb)
    max_weight = quantize_with_scaling(weight, sc, cb, H=hessian)

    sc = compute_min_mse_scaling(weight, cb, grid_size=args.grid_size)
    mse_weight = quantize_with_scaling(weight, sc, cb, H=hessian)

    sc = compute_min_mse_scaling(
        weight, cb, grid_size=args.grid_size, H=np.diag(hessian)
    )
    diag_weight = quantize_with_scaling(weight, sc, cb, H=hessian)

    sc = compute_min_mse_scaling(weight, cb, grid_size=args.grid_size, H=hessian)
    hessian_weight = quantize_with_scaling(weight, sc, cb, H=hessian)

    max_error = quantization_error(weight, max_weight, H=hessian)
    mse_error = quantization_error(weight, mse_weight, H=hessian)
    diag_error = quantization_error(weight, diag_weight, H=hessian)
    hessian_error = quantization_error(weight, hessian_weight, H=hessian)
    name = os.path.relpath(root, args.dir)
    it.write(
        f"{name}\t{max_error}\t{mse_error}\t{diag_error}\t{hessian_error}"
    )
    err_diag = diag_error / mse_error
    err_hessian = hessian_error / mse_error
    rel_error_diag.append(err_diag)
    rel_error_hessian.append(err_hessian)
    rel_error_best.append(min(err_diag, err_hessian, 1.0))

plt.plot(np.sort(rel_error_diag), label="Diagonal hessian scaling")
plt.plot(np.sort(rel_error_hessian), label="Hessian scaling")
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
