## Andrew Ashford, Jeremy Goecks lab Winter rotation project, 1/11/2021
## This Python script will read in a csv file of cyclic immunofluoresecence data, choose some columns of interest,
## randomly change a certain percent (starting at 15%) of the values of each row to 0 or use another method. This will
## then be used as inputs into an autoencoder in order to reduce the dimensionality of the data, and, hopefully allow
## for better clustering.

##### Import modules #####
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

##### Specify the Location of the CSV file #####
cycif_file_location = './HTAN2_bx2_OCT_tumor_cycif.csv'

##### Read files into their corresponding variables and preprocessing #####
# Read the file into a Pandas dataframe:
cycif_dataframe = pd.read_csv(cycif_file_location)

# Get the columns of the 25 markers only from the dataframe:
marker_cols = cycif_dataframe.filter(regex="Cell Masks$").filter(regex="^(?!(Goat|DAPI))").columns

# Read in the dataframe using only the marker columns:
cycif_dataframe_only_markers = pd.read_csv(cycif_file_location, usecols=marker_cols)

# Get the number of rows and columns of the dataframe:
num_rows, num_cols = cycif_dataframe_only_markers.shape

print('The array contains ' + str(num_cols) + ' columns and ' + str(num_rows) + ' rows.')

# Randomly shuffle the dataframe rows and separate into test and training sets:
np.random.shuffle(cycif_dataframe_only_markers.values)

# Specify percent of data you want to include in the training set:
#######################################
training_set_percent_of_data = 0.90
test_set_percent_of_data = 0.06
#######################################

# Make function to retrive training, test, and dev sets using the randomly rearranged dataframe and the percentages
# input above:
training_set = cycif_dataframe_only_markers[:int(num_rows*training_set_percent_of_data)]
test_set = cycif_dataframe_only_markers[int(num_rows*training_set_percent_of_data):int(num_rows*(training_set_percent_of_data + test_set_percent_of_data))]
validation_set = cycif_dataframe_only_markers[int(num_rows*(training_set_percent_of_data + test_set_percent_of_data)):]

# Use newer method of normalizing datasets (log10 transformation + standard scalar):
def preprocessing(dataframe):
    # Log10 transform:
    dataframe = np.log10(dataframe)
    # Drop all infinite, -infinite, and NaN values:
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
    return dataframe


processed_training_set = preprocessing(training_set)
processed_test_set = preprocessing(test_set)
processed_validation_set = preprocessing(validation_set)

# Scaling function, requires StandardScalar from Sklearn.preprocessing:
def scale_data(df_to_scale):
    # Scale training data:
    scalar = StandardScaler(with_mean=True, with_std=True)
    scalar = scalar.fit(df_to_scale)
    scaled_numpy_set = scalar.fit_transform(df_to_scale.values)
    # Change numpy array returned by StandardScalar back into a Pandas dataframe:
    scaled_pandas_set = pd.DataFrame(data=scaled_numpy_set, index=np.arange(scaled_numpy_set.shape[0]), columns=df_to_scale.columns)
    return scaled_pandas_set


# Scale all the datasets:
scaled_training_set = scale_data(processed_training_set)
scaled_test_set = scale_data(processed_test_set)
scaled_validation_set = scale_data(processed_validation_set)

print(scaled_training_set.shape)
print(scaled_test_set.shape)
print(scaled_validation_set.shape)

# Clip outliers from the data sets:
clipped_training_set = scaled_training_set[(np.abs(stats.zscore(scaled_training_set)) < 10).all(axis=1)]
clipped_test_set = scaled_test_set[(np.abs(stats.zscore(scaled_test_set)) < 10).all(axis=1)]
clipped_validation_set = scaled_validation_set[(np.abs(stats.zscore(scaled_validation_set)) < 10).all(axis=1)]

# Print the number of outlier removed from the sets:
print(str(scaled_training_set.shape[0] - clipped_training_set.shape[0]) + ' outliers were clipped from the training set of ' + str(scaled_training_set.shape[0]) + ' samples, reducing the set to ' + str(clipped_training_set.shape[0]) + ' total samples.')
print(str(scaled_test_set.shape[0] - clipped_test_set.shape[0]) + ' outliers were clipped from the test set of ' + str(scaled_test_set.shape[0]) + ' samples, reducing the set to ' + str(clipped_test_set.shape[0]) + ' total samples.')
print(str(scaled_validation_set.shape[0] - clipped_validation_set.shape[0]) + ' outliers were clipped from the validation set of ' + str(scaled_validation_set.shape[0]) + ' samples, reducing the set to ' + str(clipped_validation_set.shape[0]) + ' total samples.')

# Print the minimum and maximum value for each data frame:
# Training set min and max:
print('The training set min is: ' + str(clipped_training_set.min()))
print('The training set max is: ' + str(clipped_training_set.max()))

# Test set min and max:
print('The test set min is: ' + str(clipped_test_set.min()))
print('The test set max is: ' + str(clipped_test_set.max()))

# Validation set min and max:
print('The validation set min is: ' + str(clipped_validation_set.min()))
print('The validation set max is: ' + str(clipped_validation_set.max()))

# Generate datasets with masking noise (set a percentage of cells in the data frame to 0):
# Specify fraction of values in data frame to change:
frac = 0.15

# Noisification function:
def noisify_df(df_to_noisify):
    # Specify dataset to noisify:
    noisy_set = df_to_noisify.to_numpy()
    # Specify the number of features in dataset (should be 25 we're interested in):
    n_features = noisy_set.shape[1]
    n_samples = noisy_set.shape[0]
    # Replace the specified percentage of values in the dataframe to 0:
    k = int(frac * n_features)
    indices = np.random.randint(0, n_features, size=(n_samples, k))
    noisy_set[np.arange(n_samples)[:, None], indices] = 0
    # Change numpy array back to pandas dataframe AGAIN!:
    noisy_set = pd.DataFrame(data=noisy_set, index=np.arange(n_samples), columns=np.arange(n_features))
    return noisy_set


# Run noisification function on training set:
# Generate noisy training data:
noisy_training_set = noisify_df(clipped_training_set)

'''
# Specify the "noise factor" we want to use:
#######################################
noise_factor = 0.15
#######################################

# Generate versions of the training and test set with noise:
noisy_training_set = training_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_set.shape)
noisy_test_set = test_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_set.shape)
noisy_validation_set = validation_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=validation_set.shape)

print(noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_set.shape))

# Threshold all negative values and all values greater than 1 to 1:
noisy_training_set = np.clip(noisy_training_set, 0., 1.)
noisy_test_set = np.clip(noisy_test_set, 0., 1.)
noisy_validation_set = np.clip(noisy_validation_set, 0., 1.)
'''

# Set autoencoder parameters:
# 20, 12, 5 has yielded ~the best results so far (~50% accuracy on validation set).
#######################################
first_layer_size = 23
second_layer_size = 12
third_layer_size = 0
encoding_dimension = 8
batch_size = 10
epochs = 75
#######################################

##### Functions #####
def denoising_autoencoder(input, input_b, encoding_dimension, first_layer_size, second_layer_size, third_layer_size, batch_size, epochs):
    # Create the layers of the encoder where each "encoded" variable is a layer:
    len_input = input.shape[-1]
    input_ = Input(shape=(len_input,))
    encoded1 = Dense(units=first_layer_size, activation="relu")(input_)
    encoded2 = Dense(units=second_layer_size, activation="relu")(encoded1)
    #encoded3 = Dense(units=third_layer_size, activation="relu")(encoded2)

    # This the the layer in the middle that we want the results from:
    bottleneck = Dense(units=encoding_dimension, activation="relu")(encoded2)

    # Create the layers of the decoder where each "decoded" variable is a layer, is a mirror-image of the encoder layer:
    #decoded1 = Dense(units=third_layer_size, activation="relu")(bottleneck)
    decoded2 = Dense(units=second_layer_size, activation="relu")(bottleneck)
    decoded3 = Dense(units=first_layer_size, activation="relu")(decoded2)
    output = Dense(units=len_input, activation="linear")(decoded3)

    # Training is performed on the entire autoencoder:
    autoencoder = Model(inputs=input_, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mean_squared_error'])
    training_history = autoencoder.fit(input, input, batch_size=batch_size, epochs=epochs, validation_data=(input_b, input_b))

    # Use only the encoder part for dimensionality reduction:
    encoder = Model(inputs=input_, outputs=bottleneck)

    # Plot loss over epoch history:
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()

    # Plot accuracy over epoch history:
    plt.plot(training_history.history['acc'])
    plt.plot(training_history.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()

    return autoencoder, encoder

##### Call functions #####
autoencoder, encoder_output = denoising_autoencoder(noisy_training_set, clipped_validation_set, encoding_dimension, first_layer_size, second_layer_size, third_layer_size, batch_size, epochs)