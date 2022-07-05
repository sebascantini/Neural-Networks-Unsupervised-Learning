# =========================================== INCLUDES ===========================================
from Model import Model
import numpy as np
from matplotlib import pyplot as plt
import FileFuncions as ff

data = np.genfromtxt("data/tp2_training_dataset.csv", delimiter=",", dtype=float)[:, :]
#data = np.genfromtxt("data/home_data.csv", delimiter=",", dtype=float)[1:500,2:]
plot_name_template = "results/results_#exp_name#_plot_#run_number#.png"
exp_name = "tp2"

norm = np.linalg.norm(data)
data = data/norm

S = [851, 20, 10, 9]
model = Model(S, maxIter=2500, sanger=True, learningRate=0.2)

iters, o, learning = model.train(data)
exp_info = [model.S, model.learningRate, model.sanger, iters, model.maxIter, o]
run_number = ff.store(exp_name, exp_info)

plot_name = plot_name_template.replace("#exp_name#", exp_name)
plot_name = plot_name.replace("#run_number#", run_number)
plt.plot(learning)
plt.title("Error evolution")
plt.ylabel("Orthogonal Weights")
plt.xlabel("Iteration")
plt.savefig(plot_name)