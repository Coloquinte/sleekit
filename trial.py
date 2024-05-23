import numpy as np
from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import glob

print("Layer Base+MSE Base+CorrMSE Corr+MSE Corr+CorrMSE")
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
    err_mse = quantization_error(q_mse, weight, H)
    print(f"{err_mse}", end=" ")
    err_mse = quantization_error(q_mse, weight, removed_H)
    print(f"{err_mse}", end=" ")
    q_mse = quantize_with_scaling(weight, sc_mse, cb, H=removed_H)
    err_mse = quantization_error(q_mse, weight, H)
    print(f"{err_mse}", end=" ")
    err_mse = quantization_error(q_mse, weight, removed_H)
    print(f"{err_mse}", end=" ")
    print()
    continue

    output_scaled = apply_scaling(weight, sc_mse, 0)
    scale_input = compute_norm_scaling(output_scaled, 1)
    scaled_weight = apply_scaling(weight, scale_input, 1)
    scaled_H = np.expand_dims(scale_input, 0) * H * np.expand_dims(scale_input, 1)

    sc_scale = compute_min_mse_scaling(scaled_weight, cb, 0, min_factor=0.5, grid_size=10, H=H)
    q_scale = quantize_with_scaling(scaled_weight, sc_scale, cb, H=scaled_H)
    err_scale = quantization_error(q_scale, scaled_weight, scaled_H)
    print(f"{err_scale}", end=" ")

    output_scaled = apply_scaling(weight, sc_mse, 0)
    scale_input = compute_min_mse_scaling(output_scaled, cb, 1, min_factor=0.5, grid_size=100)
    scaled_weight = apply_scaling(weight, scale_input, 1)
    scaled_H = np.expand_dims(scale_input, 0) * H * np.expand_dims(scale_input, 1)

    sc_scale = compute_min_mse_scaling(scaled_weight, cb, 0, min_factor=0.5, grid_size=10, H=H)
    q_scale = quantize_with_scaling(scaled_weight, sc_scale, cb, H=scaled_H)
    err_scale = quantization_error(q_scale, scaled_weight, scaled_H)
    print(f"{err_scale}", end=" ")
    print()
