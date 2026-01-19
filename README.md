# MLFramework

A custom, high-performance Machine Learning Framework written in C. This framework provides a flexible neural network implementation with a command-line interface for training and inference.

## Features

- **Modular Layer System**: Supports Dense, Conv2D, Pooling, Activation (ReLU, Softmax, etc.), Dropout, and more.
- **Tensor Operations**: multidimensional tensor support with optimized backend.
- **CLI Interface**: Easy-to-use command line tools for training and inference.
- **Configurable Training**: Define network layouts and training parameters using simple text formats (`.tsl` and `.tsd`).
- **Memory Management**: Custom memory arena implementation for efficient allocation.
- **Platform Support**: Optimized for macOS (Apple Silicon/Intel) and Linux.

## Prerequisites

- CMake (3.31 or later)
- C Compiler (Clang recommended for macOS, GCC for Linux)
- Make or Ninja build system

## Build Instructions

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd MLFramework
    ```

2.  **Create a build directory**:
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake and Build**:
    ```bash
    cmake ..
    cmake --build . --target MLFramework -j 4
    ```

The executable `MLFramework` will be created in your build directory (or `cmake-build-debug` if using CLion/IDE defaults).

## Usage

The framework allows you to train models and run inference via the command line.

### 1. Training

To train a model, you need:
1.  **Network Layout (`.tsl`)**: Defines the architecture of the neural network.
2.  **Data Directory**: Directory containing your training data (e.g., MNIST `.mat` files).
3.  **Training Description (`.tsd`)**: Defines hyperparameters like epochs, batch size, optimizer, and save paths.

**Command:**
```bash
./MLFramework train <layout.tsl> <data_dir> <train_desc.tsd>
```

**Example:**
```bash
./MLFramework train tests/mnist.tsl data/mnist tests/mnist.tsd
```

### 2. Inference

To run inference using a trained model:

**Command:**
```bash
./MLFramework infer <model.tsn> <input_file>
```

**Example:**
```bash
./MLFramework infer tests/model_epoch_0001.tsn dummy_input
```
*(Note: Generic input file loading for inference is currently a placeholder. The system validates model loading.)*

## File Formats

### Network Layout (`.tsl`)
Defines the layers of the network. Example (`tests/mnist.tsl`):
```
input:
    shape = (28, 28, 1);

flatten:

dense:
    size = 128;

activation:
    type = relu;

dense:
    size = 10;

activation:
    type = softmax;
```

### Training Description (`.tsd`)
Defines training parameters. Example (`tests/mnist.tsd`):
```
epochs = 5;
batch_size = 32;
learning_rate = 0.001;
optimizer = adam;
save_interval = 1;
save_path = "tests/model_epoch_";
```

### Data Format
The implemented data loader currently supports MNIST data stored in `.mat` files (custom binary format containing raw float data).
- `train_images.mat`
- `train_labels.mat`
- `test_images.mat`
- `test_labels.mat`

## Project Structure

- `src/`: Source code
  - `layers/`: Layer implementations
  - `tensor/`: Tensor operations
  - `network/`: Network management and training logic
  - `utils/`: Data and description loaders
  - `main.c`: CLI entry point
- `include/`: Public headers
- `data/`: Dataset directory
- `tests/`: specific test configurations
