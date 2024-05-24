
from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import glob
import sys

data = []
sz = 16
cb = Codebook.uniform(sz, -1, 1)

for i in range(10):
    for weight_name in sorted(glob.glob("data/**.weight.npy")):
        name = weight_name.replace("data/model.decoder.layers.", "")
        name = name.replace(".weight.npy", "")
        if not "fc" in name and not "proj" in name:
            continue
        #print(name)
        weight = np.load(weight_name).astype(np.float32)
        hessian_name = weight_name.replace("weight", "quant_hessian")
        hessian = np.load(hessian_name).astype(np.float32)
        sc = compute_min_mse_scaling(weight, cb, grid_size=10)
        apply_scaling_in_place(weight, sc)
        data.append(np.reshape(weight, (-1,)))

    all_data = np.concatenate(data)
    cb = lloyd_max(all_data, sz, sample_count=1000)
    print(cb.values / max(abs(cb.values[0]), abs(cb.values[-1])))

    plt.hist(all_data, 10000)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.vlines(cb.values, ymin, ymax, color='red')
    plt.show()

