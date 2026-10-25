# Handwritten Digit Recognition Robot

A Python-based robotics project that uses a trained neural network to recognize handwritten digits and map predictions to movement commands for a robot in Gazebo.

## Project Overview

The project combines machine learning with robot control.

The neural network classifies handwritten digits from 0 to 9 using the MNIST dataset. The predicted digit is mapped to an action, which can be sent to a simulated robot through Gazebo's transport interface.

### Pipeline

Handwritten digit image → Neural network → Digit prediction → Action selection → Gazebo robot command

## Technologies

- Python 3
- PyTorch
- Torchvision
- MNIST dataset
- ROS 2
- Gazebo Sim
- Gazebo Transport

## Neural Network Architecture

The model is a fully connected neural network consisting of:

- Input: 784 pixel values from a 28 × 28 grayscale image
- Hidden layer: 128 neurons
- Activation: ReLU
- Output: 10 scores corresponding to digits 0–9

The model is trained using cross-entropy loss and the Adam optimizer.

## Project Structure

```text
digit_robot/
├── train.py
├── predict.py
├── find_digits.py
├── requirements.txt
├── .gitignore
├── README.md
└── digit_model.pth
```

The `data/` directory contains the downloaded MNIST dataset and is not tracked by Git.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv digit_env
source digit_env/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train.py
```

The training script downloads MNIST if necessary, trains the network, and saves the learned parameters to `digit_model.pth`.

An internet connection is required for the initial dataset download.

## Run Digit Recognition

Run:

```bash
python predict.py
```

Enter an MNIST test-image index between 0 and 9999 when prompted.

The program loads the trained weights, predicts the digit, selects a corresponding action, and attempts to send a movement command to the configured Gazebo robot.

## Digit-to-Action Mapping

| Predicted digit | Action |
|---|---|
| 0 | Stop |
| 1 | Turn left |
| 2 | Turn right |
| 3 | Move forward |
| 4 | Turn left |
| 5 | Move forward |
| 6 | Turn right |
| 7 | Move forward |
| 8 | Turn left |
| 9 | Stop |

These actions are configurable in `predict.py`.

## Gazebo Integration

The current controller targets the following Gazebo Transport topic:

```text
/model/vehicle_blue/cmd_vel
```

The Gazebo simulation and a compatible robot controller must be running before executing robot-control commands.

The topic name may need to be changed for other robot models or simulation configurations.

**Safety:** Test movement in a clear simulated environment. The example controller attempts to send a stop command after each short movement, but this is not a safety-rated control system.

## Limitations

- The current implementation selects images from the MNIST test dataset rather than using a live camera.
- The model's accuracy should be evaluated on the complete test dataset before reporting a performance figure.
- The digit-to-action mapping is a demonstration and does not provide obstacle avoidance.
- The project has been tested with a particular ROS 2/Gazebo setup; other configurations may require adjustments.

## Future Improvements

- Accept user-uploaded handwritten digit images.
- Add live camera-based digit recognition.
- Evaluate model accuracy on the MNIST test set.
- Improve robot-control reliability and safety.
- Experiment with convolutional neural networks (CNNs).

## License

Choose a license before distributing the project. If you want others to reuse and modify the code, consider an open-source license such as MIT.
