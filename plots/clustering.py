from sklearn.mixture import GaussianMixture
from pathlib import Path
import pandas as pd
from numpy import unique
from numpy import where
from matplotlib import pyplot

# iterate thorugh data cleaned data ground truth folder
data_folder = Path("data", "samples", "tumor")

biopsies: dict = {
}

# iterate through biopsies
for data in data_folder.iterdir():
    bx = "_".join(Path(data).stem.split('_')[0:3])

    if "9_" not in bx:
        continue
    print(bx)
    # load data
    bx_data = pd.read_csv(data, sep=",")
    bx_data["Biopsy"] = bx
    biopsies[bx] = bx_data

# assert that 8 unique biopsies are loaded
assert len(biopsies.keys()) == 8
marker = "CK19"

# extract all Ck19 values from all biopsies
training_data = pd.DataFrame()
for bx in biopsies:
    # add two columns, bx and value for the ck19 marker
    training_data = pd.concat([training_data, biopsies[bx][["Biopsy", marker, "X_centroid", "Y_centroid"]]])

# assert that 8 unique biopsies are loaded
assert len(training_data["Biopsy"].unique()) == 8

# reset index
training_data = training_data.reset_index(drop=True)

print("hi")
# define the model
gaussian_model = GaussianMixture(n_components=2)


# train the model
gaussian_model.fit(training_data[["Biopsy", marker]])

# assign each data point to a cluster
gaussian_result = gaussian_model.predict(training_data[["Biopsy", marker]])

# get all of the unique clusters
gaussian_clusters = unique(gaussian_result)

# plot Gaussian Mixture the clusters
for gaussian_cluster in gaussian_clusters:
    # get data points that fall in this cluster
    index = where(gaussian_result == gaussian_clusters)
    print("inex")
    print(index)
    input()

    # make the plot
    pyplot.scatter(training_data[index, "X_centroid"], training_data[index, "Y_centroid"])

# show the Gaussian Mixture plot
pyplot.show()
