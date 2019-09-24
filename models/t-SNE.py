import torch
from torch.nn import *

from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import numpy as np

l = []
with open('1.csv', 'r') as fd:
    line = fd.readline()
    while line:
        if line == "":
            continue

        line = line.strip()
        word = line.split(",")
        l.append(word)
        line = fd.readline()

data_l = DataFrame(l)
print("data_l ok")
dataMat = np.array(data_l)

pca_tsne = TSNE(n_components=2)
newMat = pca_tsne.fit_transform(dataMat)

data1 = DataFrame(newMat)
data1.to_csv('2.csv', index=False, header=False)