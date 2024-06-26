from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import os
import tqdm

import argparse


parser = argparse.ArgumentParser(
    description="Comparison of the proposed method with the standard approach"
)
parser.add_argument("dir", type=str, help="Directory containing the weights")
parser.add_argument(
    "--codebook-size", type=int, default=4, help="Size of the codebook to use"
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
args = parser.parse_args()

cb = UniformCodebook(args.codebook_size, -1, 1)

roots = sorted(
    [
        root
        for root, dirs, files in sorted(os.walk(args.dir))
        if "weight.npy" in files and "hessian.npy" in files and "mean.npy" in files
    ]
)

msg = "Data\tStandard\tCorrection\tScaling\tSleekitLight\tSleekitHeavy"

print(msg)

it = tqdm.tqdm(roots)
for root in it:
    weight = np.load(os.path.join(root, "weight.npy")).astype(np.float32)
    standard_hessian = np.load(os.path.join(root, "hessian.npy")).astype(np.float32)
    mean = np.load(os.path.join(root, "mean.npy")).astype(np.float32)
    remove_dead_values(standard_hessian, weight)
    corrected_hessian = remove_input_bias(standard_hessian, mean)
    name = os.path.relpath(root, args.dir)

    sc = compute_min_mse_scaling(
        weight,
        cb,
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    # Standard GPTQ
    standard_weight = quantize_with_scaling(
        weight, sc, cb, H=standard_hessian, act_order="diag", damp=0.01
    )
    standard_error = quantization_error(weight, standard_weight, H=standard_hessian)
    # Integrated bias correction only
    correction_weight = quantize_with_scaling(
        weight, sc, cb, H=corrected_hessian, act_order="diag", damp=0.01
    )
    correction_error = quantization_error(
        weight, correction_weight, H=corrected_hessian
    )

    sc = compute_min_mse_scaling(
        weight,
        cb,
        H=standard_hessian.diagonal(),
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    # Scaling only
    scaling_weight = quantize_with_scaling(
        weight, sc, cb, H=standard_hessian, damp=0.01
    )
    scaling_error = quantization_error(weight, scaling_weight, H=standard_hessian)

    sc = compute_min_mse_scaling(
        weight,
        cb,
        H=corrected_hessian.diagonal(),
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    # Sleekit light
    sleekit_light_weight = quantize_with_scaling(
        weight, sc, cb, H=corrected_hessian, act_order="sqerr", damp=0.03
    )
    sleekit_light_error = quantization_error(
        weight, sleekit_light_weight, H=corrected_hessian
    )

    sc = compute_obq_scaling(
        weight,
        cb,
        0,
        H=corrected_hessian,
        act_order="sqerr",
        damp=0.03,
        grid_size=args.grid_size,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    # Sleekit heavy
    sleekit_heavy_weight = quantize_with_scaling(
        weight,
        sc,
        cb,
        H=corrected_hessian,
        act_order="sqerr",
        damp=0.03,
        nb_ls_moves=100,
    )
    sleekit_heavy_error = quantization_error(
        weight, sleekit_heavy_weight, H=corrected_hessian
    )

    msg = f"{name}\t{standard_error}\t{correction_error}\t{scaling_error}\t{sleekit_light_error}\t{sleekit_heavy_error}"

    it.write(msg)
