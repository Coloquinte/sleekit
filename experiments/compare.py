from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import os
import tqdm

import argparse


parser = argparse.ArgumentParser(
    description="Comparison of the proposed method with the standard approach"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--codebook-size", type=int, default=16, help="Size of the codebook to use"
)
parser.add_argument("--damp", type=float, default=0.01, help="Hessian dampening")
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
args = parser.parse_args()

cb = Codebook.uniform(args.codebook_size, -1, 1)

roots = sorted(
    [
        root
        for root, dirs, files in sorted(os.walk(args.dir))
        if "weight.npy" in files and "hessian.npy" in files and "mean.npy" in files
    ]
)

rel_error = []

msg = "Data\tStandard\tSleekit"

print(msg)

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    standard_hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(standard_hessian, weight, pcdamp=args.damp)
    corrected_hessian = remove_input_bias(standard_hessian, mean)
    name = os.path.relpath(root, args.dir)

    sc = compute_min_mse_scaling(
        weight,
        cb,
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    standard_weight = quantize_with_scaling(weight, sc, cb, H=standard_hessian)
    standard_error = quantization_error(weight, standard_weight, H=standard_hessian)

    sc = compute_min_mse_scaling(
        weight,
        cb,
        grid_size=args.grid_size,
        H=np.diag(corrected_hessian),
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    sleekit_weight = quantize_with_scaling(weight, sc, cb, H=standard_hessian)
    sleekit_error = quantization_error(weight, sleekit_weight, H=corrected_hessian)
    rel_error.append(sleekit_error / standard_error)
    msg = f"{name}\t{standard_error}\t{sleekit_error}"

    it.write(msg)

plt.plot(np.sort(rel_error))
plt.ylim(bottom=0)

plt.title("Relative error of Sleekit compared to standard quantization")
plt.xlabel("Layers")
plt.ylabel("Relative error")

if args.save_figure is not None:
    plt.savefig(args.save_figure)
else:
    plt.show()
