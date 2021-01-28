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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
clipped_training_set = scaled_training_set[(np.abs(stats.zscore(scaled_training_set)) < 3).all(axis=1)]
clipped_test_set = scaled_test_set[(np.abs(stats.zscore(scaled_test_set)) < 3).all(axis=1)]
clipped_validation_set = scaled_validation_set[(np.abs(stats.zscore(scaled_validation_set)) < 3).all(axis=1)]

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
#######################################
frac = 0.15
#######################################

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
noisy_test_set = noisify_df(clipped_test_set)

# Distance plotting function:
def plot_dists(ax, data, title, plot_type=""):
    # Flatten data, `ravel` yields a 1D "view",
    # which is more efficient than creating a 1D copy.
    f_data = data.values.ravel()
    if plot_type == "density":
        density = stats.gaussian_kde(f_data)
        n, x, _ = plt.hist(f_data, bins=25, histtype="step", density=True)
        ax.plot(x, density(x))
    elif plot_type == "both":
        density = stats.gaussian_kde(f_data)
        n, x, _ = ax.hist(f_data, bins=25, histtype="bar", density=True)
        ax.plot(x, density(x))
    else:
        ax.hist(f_data, bins=32, histtype="bar")
    ax.set_title(title)

'''
##### Plot data distributions #####
# Generate graphs of the data distributions of all the sets before and after processing:
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), dpi=300)
plt.subplots_adjust(wspace=0.20, hspace=0.50)
plot_dists(axs[0, 0], cycif_dataframe_only_markers, "Raw Input")
plot_dists(axs[1, 0], training_set, "Raw Training Data")
plot_dists(axs[1, 1], clipped_training_set, "Normalized Training Data")
plot_dists(axs[2, 0], test_set, "Raw Test Data")
plot_dists(axs[2, 1], clipped_test_set, "Normalized Test Data")
plot_dists(axs[3, 0], validation_set, "Raw Validation Data")
plot_dists(axs[3, 1], clipped_validation_set, "Normalized Validation Data")
plt.show()
'''

# Set autoencoder parameters:
# 20, 12, 5 has yielded ~the best results so far (~50% accuracy on validation set).
# Increasing from 6->7 middle layers allows for 60% accuracy, up from ~48% accuracy.
#######################################
first_layer_size = 20
second_layer_size = 10
third_layer_size = 0
encoding_dimension = 6
batch_size = 10
epochs = 25
#######################################

##### Functions #####
def denoising_autoencoder(input, input_b, input_c, encoding_dimension, first_layer_size, second_layer_size, third_layer_size, batch_size, epochs):
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
    print(output)

    # Training is performed on the entire autoencoder:
    autoencoder = Model(inputs=input_, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mean_squared_error'])
    training_history = autoencoder.fit(input, input, batch_size=batch_size, epochs=epochs, validation_data=(input_b, input_b))
    autoencoder.summary()

    # Use only the encoder part for dimensionality reduction:
    encoder = Model(inputs=input_, outputs=bottleneck)

    # Use only the decoder part for prediction:
    decoder_input = Input(shape=(encoding_dimension,))
    next_input = decoder_input
    # get the decoder layers and apply them consecutively
    for layer in autoencoder.layers[-3:]:
        next_input = layer(next_input)
    decoder = Model(inputs=decoder_input, outputs=next_input)

    '''
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
    '''

    return autoencoder, encoder, decoder


##### Call functions #####
autoencoder, encoder_output, decoder_output = denoising_autoencoder(noisy_training_set, clipped_validation_set, clipped_test_set, encoding_dimension, first_layer_size, second_layer_size, third_layer_size, batch_size, epochs)

# Run the test set on the model, return the predicted values and use linear regression to check fit and MSE:
test_dims = clipped_test_set.shape
test_rows = test_dims[0]
test_cols = test_dims[1]
predicted_test_values_array = np.empty((0, 25), float)
for index, row in clipped_test_set.iterrows():
    current_row = row.values
    current_row = current_row.reshape(1, current_row.shape[0])
    encoded_row = encoder_output.predict(current_row)
    decoded_row = decoder_output.predict(encoded_row)
    predicted_test_values_array = np.append(predicted_test_values_array, decoded_row, axis=0)
predicted_test_values_df = pd.DataFrame(data=predicted_test_values_array, index=np.arange(clipped_test_set.shape[0]), columns=clipped_test_set.columns.tolist())

# Add both predicted and actual test values to formats that you can feed into the LinearRegression function:
predicted_test = predicted_test_values_df.values.reshape(test_rows, test_cols)
actual_test = clipped_test_set.values.reshape(test_rows, test_cols)

# Linear regression:
lr_model = LinearRegression()
lr_model.fit(predicted_test, actual_test)
mse_lr = mean_squared_error(actual_test, predicted_test)
mae_lr = mean_absolute_error(actual_test, predicted_test)
print('Mean squared error on test data: ', mse_lr)
print('Mean absolute error on test data: ', mae_lr)
print('The minimum and maximum values for the test dataset are: ' + str(clipped_test_set.min()) + ' and ' + str(clipped_test_set.max()))
print('The mean, median, and mode of the test dataset are: ' + str(clipped_test_set.mean()) + ', ' + str(clipped_test_set.median()) + ', ' + str(clipped_test_set.mode()) + ' respectively.')

# Plot function:
'''
def plot(marker_value, y_predict, title, intercept, slope):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(y_predict, marker_value, "bo")

    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = (intercept + slope * x_vals)[0]
    ax.plot(x_vals, y_vals, 'r')

    plt.xlabel("Prediction")
    plt.ylabel("Marker Value")
    plt.title(title)

    plt.show()
'''

'''
def plot(y_expected, y_predict, title):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(y_predict, y_expected, "bo")
    #plt.axline((0, 0), (1,1), color="r")
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.title(title)
    plt.show()
'''


#plot(actual_test, predicted_test, f"Linear Regression (MSE={mse_lr})")
columns = list(predicted_test_values_df.columns.values)
for i in range(0, int(predicted_test.shape[1])):
    print(columns[i])
    current_actual = []
    current_predicted = []

    for n in range(0, int(predicted_test.shape[0])):
        current_predicted.append(predicted_test[n][i])
        current_actual.append(actual_test[n][i])
    # Linear regression, MSE, and plotting:
    actual_asarray = np.asarray(current_actual, dtype=float).reshape(-1, 1)
    predicted_asarray = np.asarray(current_predicted, dtype=float).reshape(-1, 1)

    lr_model = LinearRegression()
    lr_model.fit(predicted_asarray, actual_asarray)
    intercept = lr_model.intercept_
    slope = lr_model.coef_

    mse_lr = mean_squared_error(actual_asarray, predicted_asarray)
    print('Mean squared error on test data: ', mse_lr)
    print('The minimum and maximum values for the test dataset are: ' + str(np.min(actual_asarray)) + ' and ' + str(np.max(actual_asarray)))
    print('The mean of the test dataset for the current column is: ' + str(np.mean(actual_asarray)))

    #plot(actual_asarray, predicted_asarray, (columns[i] + ' MSE = ' + str(mse_lr)))

    # predict y from the data
    x_new = np.linspace(-3, 3, 100)
    y_new = lr_model.predict(x_new[:, np.newaxis])

    # plot the results
    plt.figure(figsize=(6, 6))
    ax = plt.axes()
    ax.scatter(predicted_asarray, actual_asarray)
    ax.plot(x_new, y_new)

    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Actual values')
    ax.set_title(columns[i] + '; MSE= ' + str(mse_lr))

    ax.axis('tight')

    plt.show()
