import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import pandas
import numpy as np

for b in [1, 1.5, 2, 3]:
    try:
        data = pandas.read_csv(f"results/ordering_{b}b.csv", sep="\t")
    except FileNotFoundError:
        continue
    diag = [1.0 for d in data["Diag"]]
    diag_err = sorted(data["DiagErr"] / data["Diag"], reverse=True)
    diag_sqerr = sorted(data["DiagSqErr"] / data["Diag"], reverse=True)

    geomean_diag_err = 100 * np.exp(np.mean(np.log(diag_err))) - 100
    geomean_diag_sqerr = 100 * np.exp(np.mean(np.log(diag_sqerr))) - 100
    print(
        f"Ordering {b}b: diagonal * error {geomean_diag_err:+.2f}%, diagonal * squared error {geomean_diag_sqerr:+.2f}%"
    )

    plt.title(f"Impact of GPTQ ordering ({b}-bit); lower is better")
    plt.xlabel("Layers")
    plt.ylabel("Error relative to diagonal ordering (%)")
    plt.yscale("log")
    plt.xlim(left=0, right=len(data) - 1)
    plt.ylim(bottom=0.5, top=1.5)

    yticks = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    plt.gca().set_yticks([])
    plt.gca().set_yticks([], minor=True)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

    plt.plot(diag, label="Diagonal ordering")
    plt.plot(diag_err, label="Diagonal * error ordering")
    plt.plot(diag_sqerr, label="Diagonal * squared error ordering")
    plt.legend()
    plt.savefig(f"results/ordering_{b}b.png")
    # plt.show()
    plt.clf()

for b in [1, 1.5, 2, 3]:
    try:
        data = pandas.read_csv(f"results/correction_{b}b.csv", sep="\t")
    except FileNotFoundError:
        continue
    gptq = [1.0 for d in data["GPTQ"]]
    plus_bias = sorted(data["GPTQ+BiasCorrection"] / data["GPTQ"], reverse=True)
    with_bias = sorted(data["GPTQWithBiasCorrection"] / data["GPTQ"], reverse=True)
    geomean_plus_bias = 100 * np.exp(np.mean(np.log(plus_bias))) - 100
    geomean_with_bias = 100 * np.exp(np.mean(np.log(with_bias))) - 100
    print(
        f"Correction {b}b: plus bias {geomean_plus_bias:+.2f}%, with bias {geomean_with_bias:+.2f}%"
    )

    plt.title(f"Impact of adding bias correction ({b}-bit); lower is better")
    plt.xlabel("Layers")
    plt.ylabel("Error relative to GPTQ alone (%)")
    plt.yscale("log")
    plt.xlim(left=0, right=len(data) - 1)
    plt.ylim(bottom=0.25, top=2.0)

    yticks = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    plt.gca().set_yticks([])
    plt.gca().set_yticks([], minor=True)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

    plt.plot(gptq, label="GPTQ alone", color="blue")
    plt.plot(plus_bias, label="Bias correction after GPTQ", color="orange")
    plt.plot(with_bias, label="Bias correction during GPTQ", color="purple")
    plt.legend()
    plt.savefig(f"results/correction_{b}b.png")
    # plt.show()
    plt.clf()

for b in [1, 1.5, 2, 3]:
    try:
        data = pandas.read_csv(f"results/compare_{b}b.csv", sep="\t")
    except FileNotFoundError:
        continue
    standard = [1.0 for d in data["Standard"]]
    correction = sorted(data["Correction"] / data["Standard"], reverse=True)
    scaling = sorted(data["Scaling"] / data["Standard"], reverse=True)
    sleekit = sorted(data["Sleekit"] / data["Standard"], reverse=True)
    integrated = sorted(data["Integrated"] / data["Standard"], reverse=True)
    geomean_correction = 100 * np.exp(np.mean(np.log(correction))) - 100
    geomean_scaling = 100 * np.exp(np.mean(np.log(scaling))) - 100
    geomean_sleekit = 100 * np.exp(np.mean(np.log(sleekit))) - 100
    geomean_integrated = 100 * np.exp(np.mean(np.log(integrated))) - 100
    print(
        f"Compare {b}b: correction {geomean_correction:+.2f}%, scaling {geomean_scaling:+.2f}%, sleekit {geomean_sleekit:+.2f}%, integrated {geomean_integrated:+.2f}%"
    )

    plt.title(f"Relative error with Sleekit ({b}-bit); lower is better")
    plt.xlabel("Layers")
    plt.ylabel("Relative error (%)")
    plt.yscale("log")
    plt.xlim(left=0, right=len(data) - 1)
    plt.ylim(bottom=0.125, top=2.0)

    yticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    plt.gca().set_yticks([])
    plt.gca().set_yticks([], minor=True)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

    plt.plot(standard, label="No change", color="blue")
    plt.plot(scaling, label="Only diagonal scaling", color="green")
    plt.plot(correction, label="Only bias correction", color="orange")
    plt.plot(sleekit, label="Sleekit", color="red")
    plt.legend()
    plt.savefig(f"results/compare_{b}b.png")
    # plt.show()
    plt.clf()

for b in [1, 1.5, 2, 3]:
    try:
        data = pandas.read_csv(f"results/scaling_{b}b.csv", sep="\t")
    except FileNotFoundError:
        continue
    mse = [1.0 for d in data["MSE"]]
    diag = sorted(data["Diag"] / data["MSE"], reverse=True)
    hessian = sorted(data["Hessian"] / data["MSE"], reverse=True)
    obq = sorted(data["OBQAware"] / data["MSE"], reverse=True)
    geomean_diag = 100 * np.exp(np.mean(np.log(diag))) - 100
    geomean_hessian = 100 * np.exp(np.mean(np.log(hessian))) - 100
    geomean_obq = 100 * np.exp(np.mean(np.log(obq))) - 100
    print(
        f"Scaling {b}b: diagonal {geomean_diag:+.2f}%, hessian {geomean_hessian:+.2f}%, exhaustive {geomean_obq:+.2f}%"
    )

    plt.title(f"Impact of the scaling method ({b}-bit); lower is better")
    plt.xlabel("Layers")
    plt.ylabel("Error relative to MSE scaling (%)")
    plt.yscale("log")
    plt.xlim(left=0, right=len(data) - 1)
    plt.ylim(bottom=0.125, top=2.0)

    yticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    plt.gca().set_yticks([])
    plt.gca().set_yticks([], minor=True)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([f"{100 * (t - 1):+.0f}%" for t in yticks])

    plt.plot(mse, label="MSE scaling", color="blue")
    plt.plot(diag, label="Diagonal hessian scaling", color="green")
    plt.plot(hessian, label="Full hessian scaling", color="cyan")
    plt.plot(obq, label="Exhaustive search", color="black")
    plt.legend()
    plt.savefig(f"results/scaling_{b}b.png")
    # plt.show()
    plt.clf()
