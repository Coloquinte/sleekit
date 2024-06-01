import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import pandas
import numpy as np

data = pandas.read_csv("results/correction.csv", sep="\t")
gptq = [1.0 for d in data["GPTQ"]]
plus_bias = sorted(data["GPTQ+BiasCorrection"] / data["GPTQ"], reverse=True)
with_bias = sorted(data["GPTQWithBiasCorrection"] / data["GPTQ"], reverse=True)
best = sorted(
    np.minimum(data["GPTQ+BiasCorrection"], data["GPTQWithBiasCorrection"])
    / data["GPTQ"],
    reverse=True,
)


plt.title("Impact of adding bias correction; less is better")
plt.xlabel("Layers")
plt.ylabel("Error relative to GPTQ alone (%)")
plt.yscale("log")
plt.ylim(bottom=0.25, top=2.0)

yticks = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
plt.gca().set_yticks([])
plt.gca().set_yticks([], minor=True)
plt.gca().set_yticks(yticks)
plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

plt.plot(gptq, label="GPTQ alone")
plt.plot(plus_bias, label="Bias correction after GPTQ")
plt.plot(with_bias, label="Bias correction during GPTQ")
plt.plot(best, label="Best")
plt.legend()
plt.savefig("results/correction.png")
# plt.show()
plt.clf()

data = pandas.read_csv("results/scaling.csv", sep="\t")
mse = [1.0 for d in data["MSE"]]
diag = sorted(data["Diag"] / data["MSE"], reverse=True)
hessian = sorted(data["Hessian"] / data["MSE"], reverse=True)
obq = sorted(data["OBQAware"] / data["MSE"], reverse=True)

plt.title("Impact of the scaling method; less is better")
plt.xlabel("Layers")
plt.ylabel("Error relative to MSE scaling (%)")
plt.yscale("log")
plt.ylim(bottom=0.25, top=2.0)

yticks = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
plt.gca().set_yticks([])
plt.gca().set_yticks([], minor=True)
plt.gca().set_yticks(yticks)
plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

plt.plot(mse, label="MSE scaling")
plt.plot(diag, label="Diagonal hessian scaling")
plt.plot(hessian, label="Full hessian scaling")
plt.plot(obq, label="Exhaustive search")
plt.legend()
plt.savefig("results/scaling.png")
# plt.show()
plt.clf()

data = pandas.read_csv("results/compare.csv", sep="\t")
standard = [1.0 for d in data["Standard"]]
correction = sorted(data["Correction"] / data["Standard"], reverse=True)
scaling = sorted(data["Scaling"] / data["Standard"], reverse=True)
sleekit = sorted(data["Sleekit"] / data["Standard"], reverse=True)
integrated = sorted(data["Integrated"] / data["Standard"], reverse=True)

plt.title("Relative error with Sleekit; less is better")
plt.xlabel("Layers")
plt.ylabel("Relative error (%)")
plt.yscale("log")
plt.ylim(bottom=0.25, top=2.0)

yticks = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
plt.gca().set_yticks([])
plt.gca().set_yticks([], minor=True)
plt.gca().set_yticks(yticks)
plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

plt.plot(standard, label="No change")
plt.plot(sleekit, label="Sleekit")
plt.plot(scaling, label="Only diagonal scaling")
plt.plot(correction, label="Only bias correction")
plt.legend()
plt.savefig("results/compare.png")
# plt.show()
plt.clf()
