# CIFAR-10 Notebook: main.ipynb

This repository contains a single Jupyter Notebook, `main.ipynb`, which demonstrates a small end-to-end experiment on the CIFAR-10 dataset using TensorFlow / Keras. The notebook walks through data loading, quick visualization, simple preprocessing, and two model experiments (a dense feed-forward baseline and a small convolutional neural network). It also shows how to evaluate the final model on the test set.

## Table of contents

- Overview
- Notebook structure and sequence
  - Environment & imports
  - Data loading
  - Exploratory visualizations
  - Preprocessing / scaling
  - Dense (baseline) model
  - Convolutional Neural Network (CNN)
  - Model evaluation
- Visualizations explained
- How to run
- Notes & recommendations
- Quick interpretation of results

## Overview

`main.ipynb` is intended as a compact educational/demo notebook. It uses the CIFAR-10 dataset available from `tensorflow.keras.datasets.cifar10`. CIFAR-10 contains 60,000 32x32 color images in 10 classes (6,000 images per class), split into 50,000 training images and 10,000 test images.

The notebook is useful for:
- Learning the shape and dtype of images in CIFAR-10
- Plotting example images and their labels
- Comparing a simple dense network baseline against a small CNN
- Seeing basic preprocessing and evaluation steps

## Notebook structure and sequence

Below is the sequential breakdown of the notebook cells and what each block does.

1) Environment & imports
   - Imports: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, and TensorFlow/Keras (`tensorflow`, `keras`, `datasets`, `layers`, `models`).

2) Data loading
   - Loads CIFAR-10 with `datasets.cifar10.load_data()` into `(X_train, y_train), (X_test, y_test)`.
   - Typical shapes printed: `X_train.shape` and `X_test.shape` (expected: `(50000, 32, 32, 3)` and `(10000, 32, 32, 3)`).

3) Classes and label reshaping
   - Defines the class names array: `['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']` (note: minor spacing in the notebook's `frog ` string).
   - Reshapes `y_train` from shape `(50000, 1)` to `(50000,)` with `y_train = y_train.reshape(-1,)` so it’s easier to index/plot.

4) Exploratory visualization
   - A helper function `plot_sample(X, y, index)` is defined to display one image and its label via `plt.imshow()` and `plt.xlabel()`.
   - The notebook calls `plot_sample(X_train, y_train, 1)` twice to show an example training image.

5) Data type and scaling
   - Prints `X_train.dtype` to check the array type (typically `uint8`).
   - Scales the image pixel values to the [0,1] range: `X_train_scaled = X_train / 255` and `X_test_scaled = X_test / 255` (the notebook also divides `y_train` by 255 in one line which is incorrect—labels should not be scaled).

6) Dense (baseline) model
   - Builds a fully connected Sequential model:
     - `Flatten(input_shape=(32,32,3))`
     - `Dense(3000, activation='relu')`
     - `Dense(1000, activation='relu')`
     - `Dense(10, activation='sigmoid')`
   - Compiles with `SGD` optimizer and `sparse_categorical_crossentropy` loss.
   - Trains for 1 epoch on `X_train_scaled` and (incorrectly) `y_train_scaled` (labels were divided by 255). This is a bug: labels must be integer class indices when using `sparse_categorical_crossentropy`.

7) Convolutional Neural Network (CNN)
   - Builds a small CNN:
     - `Conv2D(32, (3,3), activation='relu')`
     - `MaxPooling2D((2,2))`
     - another `Conv2D(32, (3,3), activation='relu')` + `MaxPooling2D`
     - `Flatten()` + `Dense(64, relu)` + `Dense(10, softmax)`
   - Compiles using `adam` optimizer and `sparse_categorical_crossentropy`.
   - Trains for 10 epochs on `X_train_scaled` and `y_train_scaled` (again labels were mistakenly scaled).

8) Google Colab drive mount (optional)
   - The notebook contains a block for `from google.colab import drive` and `drive.mount('/content/drive')`. This is only relevant if running in Google Colab and can be skipped on local/other environments.

9) Model evaluation
   - Scales `X_test` (comment reminds to scale test set as well) and evaluates the CNN with `cnn.evaluate(X_test_scaled, y_test, verbose=2)` and prints test accuracy.


## Visualizations explained

- plot_sample(X, y, index)
  - Shows a single CIFAR-10 image at the requested index with its class label as the x-axis label.
  - Use this to inspect individual images for labeling sanity and class variability.

Notes on typical visuals you may add
- Grid of example images (recommended): display a 3x5 grid of images for each class or random samples. This is useful to see per-class diversity.
- Class distribution plots: bar chart of counts per class using `np.bincount(y_train)` to confirm balanced classes.

## How to run

Prerequisites
- Python 3.8+ recommended
- Install packages (example using pip):

```powershell
pip install -r requirements.txt
# or install individually
pip install tensorflow pandas matplotlib seaborn
```

Run the notebook
- Open `main.ipynb` in Jupyter, VS Code Notebook editor, or Google Colab.
- If running locally with limited GPU/CPU, reduce epochs (e.g., 1-3) or use a smaller subset of data.
- If running in Colab, you can enable GPU acceleration and optionally mount Google Drive (the notebook has a mount cell).

Suggested quick commands (PowerShell)

```powershell
# start jupyter in the repository folder
jupyter notebook
# or open the file with VS Code
code .
```

## Notes & recommendations (bugs and improvements)

- Labels must not be scaled. Replace `y_train_scaled = y_train / 255` with just `y_train` (or use one-hot encoding if switching loss to `categorical_crossentropy`). Using scaled labels breaks the loss calculation for `sparse_categorical_crossentropy` and will prevent correct training.

- Output layer activation for the dense baseline should be `softmax` when using `sparse_categorical_crossentropy`. Using `sigmoid` with 10 outputs is incorrect for multiclass classification.

- Consider using `BatchNormalization` and `Dropout` in the CNN for more stable training.

- For performance, increase dataset augmentation (e.g., random flips, crops) and a deeper CNN (or use transfer learning).

- The notebook currently trains the dense model for only 1 epoch — increase epochs to see meaningful training.

- Remove or guard Colab-specific cells when not using Colab to avoid runtime errors.

## Quick interpretation of expected results

- The simple dense network (flatten + dense) will perform poorly on CIFAR-10 compared to a convolutional model because it doesn't exploit spatial structure.

- The small CNN shown should reach reasonable accuracy (e.g., 60-75% depending on training hyperparameters and epochs). Exact results depend on training time and data augmentation.

## Final checklist

- [x] Explain the notebook structure and sequence
- [x] Explain the visualizations present and suggested improvements
- [x] Point out bugs and recommend fixes

If you'd like, I can:
- Fix the notebook issues (labels scaling & output activations), add better visualizations (image grid, class distribution), and add a `requirements.txt` plus a short test script to validate the models. Just tell me which improvements you'd like me to implement next.
