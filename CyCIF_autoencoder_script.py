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
training_set_percent_of_data = 0.92
test_set_percent_of_data = 0.06
#######################################

# Retrive training, test, and dev sets using the randomly rearranged dataframe and the percentages input above:
training_set = cycif_dataframe_only_markers[:int(num_rows*training_set_percent_of_data)]
test_set = cycif_dataframe_only_markers[int(num_rows*training_set_percent_of_data):int(num_rows*(training_set_percent_of_data + test_set_percent_of_data))]
validation_set = cycif_dataframe_only_markers[int(num_rows*(training_set_percent_of_data + test_set_percent_of_data)):]

# Normalize training and test sets:
training_set = training_set/np.max(training_set)
test_set = test_set/np.max(test_set)
validation_set = validation_set/np.max(validation_set)

# Specify the "noise factor" we want to use:
#######################################
noise_factor = 0.5
#######################################

# Generate versions of the training and test set with noise:
noisy_training_set = training_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_set.shape)
noisy_test_set = test_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_set.shape)
noisy_validation_set = validation_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=validation_set.shape)

# Threshold all negative values and all values greater than 1 to 1:
noisy_training_set = np.clip(noisy_training_set, 0., 1.)
noisy_test_set = np.clip(noisy_test_set, 0., 1.)
noisy_validation_set = np.clip(noisy_validation_set, 0., 1.)

# Set autoencoder parameters:
#######################################
first_layer_size = 20
second_layer_size = 15
third_layer_size = 10
encoding_dimension = 6
batch_size = 10
epochs = 350
#######################################

##### Functions #####
def denoising_autoencoder(input, input_b, encoding_dimension, first_layer_size, second_layer_size, third_layer_size, batch_size, epochs):
    # Create the layers of the encoder where each "encoded" variable is a layer:
    len_input = input.shape[-1]
    input_ = Input(shape=(len_input,))
    encoded1 = Dense(units=first_layer_size, activation="relu")(input_)
    encoded2 = Dense(units=second_layer_size, activation="relu")(encoded1)
    encoded3 = Dense(units=third_layer_size, activation="relu")(encoded2)

    # This the the layer in the middle that we want the results from:
    bottleneck = Dense(units=encoding_dimension, activation="relu")(encoded3)

    # Create the layers of the decoder where each "decoded" variable is a layer, is a mirror-image of the encoder layer:
    decoded1 = Dense(units=third_layer_size, activation="relu")(bottleneck)
    decoded2 = Dense(units=second_layer_size, activation="relu")(decoded1)
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
autoencoder, encoder_output = denoising_autoencoder(noisy_training_set, noisy_validation_set, encoding_dimension, first_layer_size, second_layer_size, third_layer_size, batch_size, epochs)