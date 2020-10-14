
import io
import argparse
import os
import keys
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math


def read_arguments():
    """
    reads the dataset path from the console
    :param: none
    :return: the dataset path
    """

    # set a usage to show how to run the tool
    usage = '%(prog)s [-h] --dataset_path'

    # set a description about this tool
    description = 'this tool is used for building a linear regression model from the passed dataset'

    # create the parser and set the usage and the description
    parser = argparse.ArgumentParser(description=description, usage=usage)

    # define an argument to be passed to parse and get it
    parser.add_argument('-path', dest='dataset_path', type=str, required=True,
                        help='the dataset path')

    # get the argument which is the dataset path
    args = parser.parse_args()

    # return the dataset path
    return args.dataset_path


def write_parameters_into_json(parameters, dir_path):
    """write the linear regression parameters into Json file

    Args:
        parameters (dict): a dict contains theta0 and theta1
        dir_path (string): directory path to save the file on it
    """

    # prepare the enire path of the file
    json_file_name = "parameters.json"
    json_file_with_path = os.path.join(dir_path, json_file_name)

    # create the file and write the parameters on it
    with open(json_file_with_path, "w+") as outfile:
        outfile.write(json.dumps(parameters, indent=4))


def save_linear_regression_model_into_image(parameters, feature_dataset,
                                            target_dataset, dir_path):
    """save linear regression model plot into png image

    Args:
        parameters (dict): a dict contains theta0 and theta1
        dir_path (string): directory path to save the image on it
    """
    # create a range of values between the min and max values of the features dataset
    x_axis_range = np.linspace(min(feature_dataset), max(feature_dataset), 100)

    # define the linear regression model which is a list of predicted values of the x_axis_range
    predicted_target_dataset = parameters.get(
        keys.THETA0) + parameters.get(keys.THETA1) * x_axis_range

    # resize the plot
    plt.figure(figsize=(18, 8))

    # set titles and labels
    plt.title("Representation Of The Data And The Linear Regression Model")
    plt.xlabel("feature")
    plt.ylabel("target")

    # plotting the original data points
    plt.scatter(feature_dataset, target_dataset, label="Original Data")

    # plotting the linear regression model
    plt.plot(x_axis_range, predicted_target_dataset,
             color="red", label="Linear Regression Model")

    # add text to explain the value of alfa
    plt.annotate("Learning Rate = "+str(config.LEARNING_RATE), xy=(111.5, 0))
    plt.legend()

    # define the name of the image
    image_file_name = 'result.png'

    # prepare the image path with its name
    image_file_with_path = os.path.join(dir_path, image_file_name)

    # save the image to png image
    plt.savefig(image_file_with_path)


def is_convarge(old_theta0, old_theta1, new_theta0, new_theta1):
    """check if the distance between the two points (old_theta0,old_theta1) and
        (new_theta0,new_theta1) is small or not to make sure that there is a convergence

    Args:
        new_theta0 (float): new value of theta0
        old_theta0 (float): old value of theta0
        new_theta1 (float): new value of theta1
        old_theta1 (float): old value of theta1

    Returns:
        boolean: if there is a convergence or not
    """

    # return the test result as boolean value
    return (math.sqrt((old_theta0 - new_theta0)**2
                      + (old_theta1 - new_theta1)**2) <= config.THRESHOLD)


def gradient_descent(feature_dataset, target_dataset, data_size):
    """find linear regression parameters (theta0 and theta1)

    Args:
        feature_dataset (list of floats): feature data list
        target_dataset (list of floats): target data list

    Returns:
        tuple : the linear regression parameters (theta0 and theta1)
    """

    # get the initial values of the parameters and store them in a dictionary
    parameters = {
        keys.THETA0: config.THETA0_INITIAL,
        keys.THETA1: config.THETA1_INITIAL
    }

    # get the learning rate and store it
    learning_rate = config.LEARNING_RATE

    # normalizatin factor
    norm_factor = (1/data_size)

    # repeat until convergence
    while(True):

        # define the linear regression model which is a list of predicted values of the features
        target_predicted = parameters.get(keys.THETA0) + \
            feature_dataset * parameters.get(keys.THETA1)

        # get the change on theta0
        changes_on_theta0 = learning_rate * norm_factor * \
            np.sum(target_predicted - target_dataset)

        # get the change on theta1
        changes_on_theta1 = learning_rate * norm_factor * \
            np.sum((target_predicted - target_dataset) * feature_dataset)

        # get the new value of theta0
        theta0_changed = parameters.get(keys.THETA0) - changes_on_theta0

        # get the new value of theta1
        theta1_changed = parameters.get(keys.THETA1) - changes_on_theta1

        # check if we reach convergence or not
        if(is_convarge(parameters.get(keys.THETA0), parameters.get(keys.THETA1),
                       theta0_changed, theta1_changed)):
            break

        # edit the values of thetas if they passed the convergence test
        parameters[keys.THETA1] = theta1_changed
        parameters[keys.THETA0] = theta0_changed

    # return the parameters
    return parameters


def read_dataset(dataset_path):
    """read the fearures and the target datasets from the file

    Args:
        file_path (string): file path to the dataset

    Returns:
        tuple: fearures and the target datasets
    """

    # read the dataset into dataframe
    dataset = pd.read_excel(dataset_path)

    # slice the feature and the target lists
    feature_dataset = np.array(dataset.X)
    target_dataset = np.array(dataset.Y)

    # return the feature_dataset and target_dataset
    return feature_dataset, target_dataset


def main():

    # read the dataset path from the console
    dataset_path = read_arguments()

    # check if the file exists or not
    if not os.path.isfile(dataset_path):
        print("Error. Please make sure the dataset file is exist!")
        return

    # slice the directory path
    dir_path = os.path.dirname(dataset_path)

    # read the dataset and get the feature and the target datasets
    feature_dataset, target_dataset = read_dataset(dataset_path)

    # get the dataset size
    data_size = len(feature_dataset)

    # fit the model using gredient descent and get the parameters (theta0 and theta1)
    parameters = gradient_descent(
        feature_dataset, target_dataset, float(data_size))

    # print the parameters
    print(parameters)

    # write the parameters into Json file
    write_parameters_into_json(parameters, dir_path)

    # save the plot of the linear regression model into png image
    save_linear_regression_model_into_image(
        parameters, feature_dataset, target_dataset, dir_path)


# to make sure that the code will be executed just when run it not import it
if __name__ == "__main__":
    main()
