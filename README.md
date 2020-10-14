## Introduction

This tool is used for building a linear regression model using the gradient descent algorithm.

## Algorithm behind

The tool takes as a parameter the dataset path. We used this dataset:

`https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr06.html`

- The program searches for the parameters of the linear function using gradient descent.

- The initial values of the parameters are zeros.

- We explained in the Jupyter Notebook file how we choose alpha.

- The tool at the end prints the optimal values of the parameters into the screen. And save them into JSON file in the dataset directory.

- The tool saves an image containing a plot of the data and the linear function that was computed in the dataset directory.

## How to use

#### to run the tool:

- run this command `python3 linear_model.py -path dataset_path`
  - where the `dataset_path` is the path for the dataset
