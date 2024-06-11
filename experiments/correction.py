from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import os
import tqdm

import argparse

parser = argparse.ArgumentParser(
    description="Analysis of the effect of bias correction on the error"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--codebook-size", type=int, default=4, help="Size of the codebook to use"
)
parser.add_argument("--damp", type=float, default=0.0001, help="Hessian dampening")

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

print("Data\tScaling\tGPTQ\tGPTQ+BiasCorrection\tGPTQWithBiasCorrection")
it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(hessian, weight, damp=args.damp)
    corrected_hessian = remove_input_bias(hessian, mean)

    sc = compute_scaling(
        weight,
        cb,
        H=hessian,
        mode=args.scaling,
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )

    gptq_weight = quantize_with_scaling(weight, sc, cb, H=hessian)
    gptq_with_bias_weight = quantize_with_scaling(weight, sc, cb, H=corrected_hessian)

    gptq_error = quantization_error(weight, gptq_weight, H=hessian)
    gptq_plus_bias_error = quantization_error(weight, gptq_weight, H=corrected_hessian)
    gptq_with_bias_error = quantization_error(
        weight, gptq_with_bias_weight, H=corrected_hessian
    )
    name = os.path.relpath(root, args.dir)
    it.write(
        f"{name}\t{args.scaling}\t{gptq_error}\t{gptq_plus_bias_error}\t{gptq_with_bias_error}"
    )
