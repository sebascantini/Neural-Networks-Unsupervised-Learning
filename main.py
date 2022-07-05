from operator import mod
from statistics import mode
from Model import Model
import numpy as np

data = np.genfromtxt("data/tp2_training_dataset.csv", delimiter=",", dtype=float)[:, :]

norm = np.linalg.norm(data)
data = data/norm

S = [851, 15, 9]
model = Model(S, maxIter=1000, sanger=False)
iters, o = model.train(data)