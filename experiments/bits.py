from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import os
import tqdm

import argparse

parser = argparse.ArgumentParser(
    description="Analysis of the effect of the number of bits on the error"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument("--damp", type=float, default=0.01, help="Hessian dampening")
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

roots = sorted(
    [
        root
        for root, dirs, files in sorted(os.walk(args.dir))
        if "weight.npy" in files and "hessian.npy" in files and "mean.npy" in files
    ]
)

bits = [
    (2, 1),
    (3, 1.5),
    (4, 2),
    (5, 2.3),
    (7, 2.8),
    (8, 3),
    (9, 3.2),
    (15, 3.9),
    (16, 4),
    (32, 5),
]

msg = "Data"
for sz, b in bits:
    msg += f"\tStandard{b}-bit"
for sz, b in bits:
    msg += f"\tSleekitLight{b}-bit"
print(msg)

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight)
    name = os.path.relpath(root, args.dir)
    msg = f"{name}"

    for sz, b in bits:
        cb = UniformCodebook(sz, -1, 1)
        sc = compute_scaling(
            weight,
            cb,
            H=hessian,
            mode="mse",
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
        quant_weight = quantize_with_scaling(
            weight, sc, cb, H=hessian, act_order="diag", damp=0.01
        )
        error = quantization_error(weight, quant_weight, H=hessian)
        msg += f"\t{error}"

    for sz, b in bits:
        cb = UniformCodebook(sz, -1, 1)
        sc = compute_scaling(
            weight,
            cb,
            H=hessian,
            mode="diag",
            grid_size=args.grid_size,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
        # FIXME: this should use the corrected hessian here
        quant_weight = quantize_with_scaling(
            weight, sc, cb, H=hessian, act_order="sqerr", damp=0.03
        )
        error = quantization_error(weight, quant_weight, H=hessian)
        msg += f"\t{error}"

    it.write(msg)
