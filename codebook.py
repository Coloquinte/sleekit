
from sleekit.codebook import *
from sleekit.obq import *
from sleekit.scaling import *
import matplotlib.pyplot as plt
import glob
import sys

data = []
for weight_name in sorted(glob.glob("data/**.weight.npy")):
    name = weight_name.replace("data/model.decoder.layers.", "")
    name = name.replace(".weight.npy", "")
    if not "fc" in name and not "proj" in name:
        continue
    print(name)
    weight = np.load(weight_name).astype(np.float32)
    hessian_name = weight_name.replace("weight", "quant_hessian")
    hessian = np.load(hessian_name).astype(np.float32)
    sc = compute_norm_scaling(weight)
    apply_scaling_in_place(weight, sc)
    sc = compute_norm_scaling(weight, 1)
    coeff = np.corrcoef(np.diag(hessian), np.square(sc))[0, 1]
    print(f"Coefficient: {coeff}")
    data.append(np.reshape(weight, (-1,)))
    #plt.hist(np.reshape(weight, (-1,)), 100)
    #plt.show()

all_data = np.concatenate(data)
plt.hist(all_data, 1000)
plt.show()
sys.exit(0)

for lagrange_mult in [0.0, 0.001]:
    cb = lloyd_max(all_data, 256, lagrange_mult)
    print(cb.values)
    print(cb.entropy(all_data))

