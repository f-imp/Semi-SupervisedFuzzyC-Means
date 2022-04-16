import os
import random
import numpy as np
import cupy as cp
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import warnings

from implementation.for_loops import ssfcm_v1
from implementation.matrix_multiplication import ssfcm_v2
from implementation.matrix_multiplication_gpu import ssfcm_v3
from utils.funcs import create_dataset_partially_labelled_new, get_b_F, create_clustering
from utils.visualization import visualizing_clustering_result

warnings.filterwarnings("ignore")
import shutil

path_result = os.path.join(os.getcwd(), "Result")
os.makedirs(name=path_result, exist_ok=False)

seed = 42
number_of_samples = [100, 1000, 10000]
number_of_features = [10, 100, 500]
number_of_clusters = [3, 5, 10]

percentage_of_labelled_samples = 20

info = {"version": [], "n_samples": [], "n_features": [], "n_clusters": [], "time [s]": []}

cont = 0
for s in number_of_samples:
    for f in number_of_features:
        for c in number_of_clusters:
            random.seed(seed)
            np.random.seed(seed)

            print("\nExperiment:=> Samples: {:>6} - Features: {:>3} - Clusters: {:>3}".format(s, f, c))
            path_exp = os.path.join(path_result, str(s) + "_" + str(f) + "_" + str(c))
            os.makedirs(name=path_exp, exist_ok=False)

            X_raw, y_raw = make_blobs(n_samples=s, n_features=f, centers=c, random_state=seed)
            print(np.unique(ar=y_raw, return_counts=True))

            X, y, a, n = create_dataset_partially_labelled_new(X_raw, y_raw, percentage_of_labelled_samples, seed)
            indexes = np.random.randint(low=0, high=np.shape(X)[0], size=int(np.shape(X)[0] / 100 * 50))
            b, F = get_b_F(y=y, n_clusters=c)

            name_image = os.path.join(path_exp, "Original_" + str(s) + "_" + str(f) + "_" + str(c) + ".png")
            visualizing_clustering_result(cont, [X_raw, y_raw], seed=seed, title="\nOriginal", filepath=name_image,
                                          n_cluster=c, idx=indexes)
            cont += 1

            U_v1, v_v1, time_v1 = ssfcm_v1(X=X, number_of_clusters=c,
                                           fuzziness_coefficient=2,
                                           b=b, F=F, alpha=a,
                                           distance='euclidean')
            info["n_samples"].append(s)
            info["n_features"].append(f)
            info["n_clusters"].append(c)
            info["version"].append("for-loops (v1)")
            info["time [s]"].append(time_v1)
            data_clustered_v1, y_v1 = create_clustering(X=X, U=U_v1)
            name_image = os.path.join(path_exp, "SSFCM_v1_" + str(s) + "_" + str(f) + "_" + str(c) + ".png")
            title = "\nSSFCM - V1 Samples: " + str(s) + " Features: " + str(f) + " Clusters: " + str(c)
            visualizing_clustering_result(cont, [X, y_v1], seed=seed, title=title, filepath=name_image, n_cluster=c,
                                          idx=indexes)
            cont += 1

            U_v2, v_v2, time_v2 = ssfcm_v2(X=X, number_of_clusters=c,
                                           fuzziness_coefficient=2,
                                           b=b, F=F, alpha=a,
                                           distance='euclidean')
            info["n_samples"].append(s)
            info["n_features"].append(f)
            info["n_clusters"].append(c)
            info["version"].append("matrix (v2)")
            info["time [s]"].append(time_v2)
            data_clustered_v2, y_v2 = create_clustering(X=X, U=U_v2)
            name_image = os.path.join(path_exp, "SSFCM_v2_" + str(s) + "_" + str(f) + "_" + str(c) + ".png")
            title = "\nSSFCM - V2 Samples: " + str(s) + " Features: " + str(f) + " Clusters: " + str(c)
            visualizing_clustering_result(cont, [X, y_v2], seed=seed, title=title, filepath=name_image, n_cluster=c,
                                          idx=indexes)
            cont += 1

            U_v3, v_v3, time_v3 = ssfcm_v3(X=cp.asarray(X), number_of_clusters=c,
                                           fuzziness_coefficient=2,
                                           b=cp.asarray(b), F=cp.asarray(F), alpha=a,
                                           distance='euclidean')
            U_v3, v_v3 = cp.asnumpy(U_v3), cp.asnumpy(v_v3)
            info["n_samples"].append(s)
            info["n_features"].append(f)
            info["n_clusters"].append(c)
            info["version"].append("matrix GPU(v3)")
            info["time [s]"].append(time_v3)
            data_clustered_v3, y_v3 = create_clustering(X=X, U=U_v3)
            name_image = os.path.join(path_exp, "SSFCM_v3_" + str(s) + "_" + str(f) + "_" + str(c) + ".png")
            title = "\nSSFCM - V3 Samples: " + str(s) + " Features: " + str(f) + " Clusters: " + str(c)
            visualizing_clustering_result(cont, [X, y_v3], seed=seed, title=title, filepath=name_image, n_cluster=c,
                                          idx=indexes)
            cont += 1
            print("\n{:<35} {:>10}[s]\n{:<35} {:>10}[s]\n{:<35} {:>10}[s]".format("For-Loops (v1)",
                                                                                  np.round(time_v1, decimals=5),
                                                                                  "Matrix-Multiplication (v2)",
                                                                                  np.round(time_v2, decimals=5),
                                                                                  "Matrix-Multiplication (v3)",
                                                                                  np.round(time_v3, decimals=5)))
df = pd.DataFrame(info)
df.to_csv(os.path.join(path_result, "report.csv"))

g = sns.catplot(x="n_clusters", y="time [s]",
                hue="version",
                col="n_features",
                row="n_samples",
                data=df, kind="bar", sharex=False, height=5, aspect=.9)
g.savefig(os.path.join(path_result, 'output.png'))

df2 = df.drop(df[df.version == "for-loops (v1)"].index)
g2 = sns.catplot(x="n_clusters", y="time [s]",
                 hue="version",
                 col="n_features",
                 row="n_samples",
                 data=df2, kind="bar", sharex=False, sharey=True, height=5, aspect=.9)
g2.savefig(os.path.join(path_result, 'output2.png'))

df3 = df2.drop(df2[df2.version == "matrix (v2)"].index)
g3 = sns.catplot(x="n_clusters", y="time [s]",
                 hue="version",
                 col="n_features",
                 row="n_samples",
                 data=df3, kind="bar", sharex=False, sharey=True, height=5, aspect=.9)
g3.savefig(os.path.join(path_result, 'output3.png'))

shutil.make_archive(path_result.split("/")[-1], 'zip', path_result)
