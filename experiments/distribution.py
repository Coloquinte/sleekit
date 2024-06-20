from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import os
import tqdm

import argparse

parser = argparse.ArgumentParser(
    description="Show the distribution of weight values after scaling"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--scaling",
    type=str,
    choices=["norm", "max", "mse", "hessian", "obq"],
    default="norm",
    help="Scaling method to use",
)
parser.add_argument(
    "--codebook-size",
    type=int,
    default=16,
    help="Size of the codebook for error minimization",
)
parser.add_argument(
    "--grid-size", type=int, default=100, help="Grid size for error minimization"
)
parser.add_argument("--damp", type=float, default=0.01, help="Hessian dampening")
parser.add_argument("--save-figure", type=str, help="Save the figure to this file")
parser.add_argument("--save-data", type=str, help="Save the numpy data to this file")
args = parser.parse_args()

data = []
cb = UniformCodebook(args.codebook_size, -1, 1)

roots = sorted(
    [
        root
        for root, dirs, files in sorted(os.walk(args.dir))
        if "weight.npy" in files and "hessian.npy" in files
    ]
)

for root in tqdm.tqdm(roots):
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    remove_dead_values(hessian, weight)
    if args.scaling == "norm":
        sc = compute_norm_scaling(weight)
    elif args.scaling == "max":
        sc = compute_non_saturating_scaling(weight, cb)
    elif args.scaling == "hessian":
        sc = compute_min_mse_scaling(weight, cb, H=hessian, grid_size=args.grid_size)
    elif args.scaling == "obq":
        sc = compute_obq_scaling(weight, cb, H=hessian, grid_size=args.grid_size)
    elif args.scaling == "mse":
        sc = compute_min_mse_scaling(weight, cb, grid_size=args.grid_size)
    else:
        raise RuntimeError(f"Unknown scaling {args.scaling}")
    apply_scaling_in_place(weight, sc)
    data.append(np.reshape(weight, (-1,)))

all_data = np.concatenate(data)
if args.save_data is not None:
    np.save(args.save_data, all_data)

plt.title(f"Weights after {args.scaling} scaling")
plt.xlabel("Value")
plt.ylabel("Density")
plt.hist(all_data, 1000, density=True)
if args.save_figure is not None:
    plt.savefig(args.save_figure)
else:
    plt.show()
