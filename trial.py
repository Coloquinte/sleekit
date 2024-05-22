import numpy as np
from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import glob

print("Layer MSE MSE+corr H H+corr OBQ OBQ+corr")
for hessian_name in sorted(glob.glob("data/**.quant_hessian.npy")):
    name = hessian_name.replace("model.decoder.layers.", "")
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
    sc_mse = compute_min_mse_scaling(weight, cb, 0, min_factor=0.5, grid_size=10)
    q_mse = quantize_with_scaling(weight, sc_mse, cb, H=H)
    err_mse = quantization_error(q_mse, weight, removed_H)
    print(f"{err_mse}", end=" ")
    q_mse_corr = quantize_with_scaling(weight, sc_mse, cb, H=removed_H)
    err_mse_corr = quantization_error(q_mse_corr, weight, removed_H)
    print(f"{err_mse_corr}", end=" ")
    sc_hessian = compute_min_mse_scaling(
        weight, cb, 0, H=H, min_factor=0.5, grid_size=10
    )
    q_hessian = quantize_with_scaling(weight, sc_hessian, cb, H=H)
    err_hessian = quantization_error(q_hessian, weight, removed_H)
    print(f"{err_hessian}", end=" ")
    q_hessian_corr = quantize_with_scaling(weight, sc_hessian, cb, H=removed_H)
    err_hessian_corr = quantization_error(q_hessian_corr, weight, removed_H)
    print(f"{err_hessian_corr}", end=" ")
    sc_obq = compute_min_mse_scaling(
        weight, cb, 0, H=H, obq=True, min_factor=0.5, grid_size=10
    )
    q_obq = quantize_with_scaling(weight, sc_obq, cb, H=H)
    err_obq = quantization_error(q_obq, weight, removed_H)
    print(f"{err_obq}", end=" ")
    q_obq_corr = quantize_with_scaling(weight, sc_obq, cb, H=removed_H)
    err_obq_corr = quantization_error(q_obq_corr, weight, removed_H)
    print(f"{err_obq_corr}", end=" ")
    print()
