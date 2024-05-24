import numpy as np
from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import glob
import sys

print("Layer MSE1 MSE4 MSE5 MSE6 H1 H4 H5 H6")
for hessian_name in sorted(glob.glob("data/**.quant_hessian.npy")):
    name = hessian_name.replace("data/model.decoder.layers.", "")
    name = name.replace(".quant_hessian.npy", "")
    print(f"{name}", end=" ")
    H = np.load(hessian_name)
    weight = np.load(hessian_name.replace("quant_hessian", "weight")).astype(np.float32)
    mean = np.load(hessian_name.replace("quant_hessian", "quant_mean"))

    damp = 1.0e-4
    mean_diag = np.mean(np.diag(H))

    # Remove dead elements like GPTQ
    dead = np.diag(H) == 0
    H[dead, dead] = mean_diag
    weight[:, dead] = 0

    H += np.diag(np.full(H.shape[0], damp * mean_diag))
    removed_H = remove_input_bias(H, mean)

    cb = Codebook.uniform(17, -8, 8)
    for hessian in [None, H]:
        scale = compute_min_mse_scaling(
            weight, cb, 0, min_factor=0.5, grid_size=20, H=hessian
        )
        for order in [1, 4, 5, 6]:
            q_weight = quantize_with_scaling(weight, scale, cb, H=H, act_order=order)
            err = quantization_error(q_weight, weight, H)
            print(f"{err}", end=" ")
            sys.stdout.flush()
    print()
