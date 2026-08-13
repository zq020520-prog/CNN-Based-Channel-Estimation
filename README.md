# CNN-based Wireless Channel Estimation

A deep learning-based wireless channel estimation method using a Convolutional Neural Network (CNN).

## Overview

This project investigates the application of **Convolutional Neural Networks (CNNs)** to wireless channel estimation.

The received pilot signals are used as the input to the neural network, and the CNN learns the relationship between the received signal and the underlying wireless channel. The trained model can then estimate the channel from the received pilot signal.

Compared with conventional channel estimation methods, the CNN-based approach aims to improve channel estimation performance under different noise and SNR conditions.

## System Model

The wireless communication system can be modeled as:

```text
        Transmitter
             │
             ▼
        Pilot Signal
             │
             ▼
      Wireless Channel
             │
             ▼
           Noise
             │
             ▼
        Received Signal
             │
             ▼
          CNN Model
             │
             ▼
      Estimated Channel
```

For a flat-fading channel, the received signal can be represented as:

```text
y = h · x + n
```

where:

* `x` — transmitted pilot signal
* `h` — wireless channel coefficient
* `n` — additive noise
* `y` — received signal

The CNN learns to estimate `h` from the received signal `y`.

## Method

The overall processing procedure is:

1. Generate wireless channel samples.
2. Generate pilot signals.
3. Transmit the pilot signals through the simulated wireless channel.
4. Add noise under different SNR conditions.
5. Construct the training and testing datasets.
6. Train the CNN model.
7. Use the trained CNN to estimate the channel.
8. Evaluate the estimation performance.

## CNN Architecture

The CNN consists of convolutional layers followed by nonlinear activation and fully connected layers.

```text
Received Signal
      │
      ▼
Convolution
      │
      ▼
Activation
      │
      ▼
Convolution
      │
      ▼
Fully Connected
      │
      ▼
Estimated Channel
```

The network is trained by minimizing the difference between the estimated channel and the true channel.

## Performance Evaluation

The channel estimation performance can be evaluated using metrics such as **Mean Squared Error (MSE)** and **Normalized Mean Squared Error (NMSE)**.

The performance is evaluated under different SNR conditions to investigate the robustness of the CNN-based channel estimator.

## Technologies

* Python
* CNN
* Deep Learning
* Wireless Channel Modeling
* Channel Estimation
* SNR-based Performance Evaluation

## Project Structure

```text
.
├── data/              # Training and testing data
├── model/             # CNN model
├── train/             # Model training
├── test/              # Model testing
├── results/           # Experimental results
└── README.md
```

## Results

The trained CNN model is evaluated under different SNR conditions.

Performance can be visualized using:

* MSE / NMSE versus SNR
* Estimated channel versus true channel
* Training loss curve
* CNN-based estimation versus conventional methods


## License

This project is for research and educational purposes.
