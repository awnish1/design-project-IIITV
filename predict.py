
import torch
from torch import nn
from torchvision import datasets, transforms
import subprocess
import time

# Gazebo velocity command topic for the blue robot
ROBOT_TOPIC = "/model/vehicle_blue/cmd_vel"


# 1. Define the neural network architecture
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# 2. Load the trained neural network weights
model = DigitNet()

model.load_state_dict(
    torch.load(
        "digit_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# 3. Load the MNIST test dataset
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# 4. Map each predicted digit to a robot action
def choose_action(digit):
    actions = {
        0: "STOP",
        1: "TURN_LEFT",
        2: "TURN_RIGHT",
        3: "MOVE_FORWARD",
        4: "TURN_LEFT",
        5: "MOVE_FORWARD",
        6: "TURN_RIGHT",
        7: "MOVE_FORWARD",
        8: "TURN_LEFT",
        9: "STOP"
    }

    return actions[digit]


# 5. Publish a velocity command to Gazebo
def send_velocity(linear_x, angular_z):
    message = (
        f"linear: {{x: {linear_x}}}, "
        f"angular: {{z: {angular_z}}}"
    )

    subprocess.run(
        [
            "gz",
            "topic",
            "-t", ROBOT_TOPIC,
            "-m", "gz.msgs.Twist",
            "-p", message
        ],
        check=True,
        timeout=5
    )


# 6. Execute the selected robot action
def move_robot(action):
    commands = {
        "STOP": (0.0, 0.0),
        "TURN_LEFT": (0.0, 0.5),
        "TURN_RIGHT": (0.0, -0.5),
        "MOVE_FORWARD": (0.2, 0.0)
    }

    linear_x, angular_z = commands[action]

    try:
        if action == "STOP":
            send_velocity(0.0, 0.0)
            print("Robot stopped.")
            return

        print(f"Executing action: {action}")

        # Start the movement
        send_velocity(linear_x, angular_z)

        # Allow a short movement interval
        time.sleep(1.0)

    except (subprocess.SubprocessError, FileNotFoundError) as error:
        print("Could not control the robot:", error)

    finally:
        # Attempt to stop after the action
        try:
            send_velocity(0.0, 0.0)
            print("Stop command sent.")
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            print("WARNING: Could not stop robot:", error)


# 7. Ask the user to select an MNIST test image
index = int(input("Enter test image index (0-9999): "))

if not 0 <= index < len(test_data):
    raise ValueError("Index must be between 0 and 9999.")


image, actual_digit = test_data[index]


# 8. Predict the digit using the trained weights
with torch.no_grad():
    output = model(image.unsqueeze(0))
    probabilities = torch.softmax(output, dim=1)

    predicted_digit = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()


# 9. Display the prediction
print("Actual digit:", actual_digit)
print("Predicted digit:", predicted_digit)
print(f"Confidence: {confidence:.2%}")


# 10. Select the robot action
action = choose_action(predicted_digit)

print("Recognized digit:", predicted_digit)
print("Selected robot action:", action)


# 11. Execute the action in Gazebo
move_robot(action)

