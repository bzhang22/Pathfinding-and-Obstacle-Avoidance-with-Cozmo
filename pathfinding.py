import map_creation
import cozmo
from cozmo.util import degrees, distance_inches, speed_mmps
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import asyncio

mymap = map_creation.Map(8, 8)
mymap.add_street(0, 6, 6, 6)
mymap.add_street(0, 0, 7, 0)

mymap.add_street(6, 0, 6, 7)
mymap.set_current_point(7, 0)
mymap.set_destination(1, 2)

path = mymap.a_star_search()

mymap.display_map(path=path)
print("Path found:", path)


def on_qr_code_detected(evt, **kwargs):
    # Cozmo detected a QR code
    print("QR Code detected: ", evt.code.data)
    robot.say_text(f"I found a QR code. It says: {evt.code.data}").wait_for_completed()

import cozmo
import asyncio
import time

def measure_distance_to_cube(robot):
    try:
        cube = robot.world.wait_for_observed_light_cube(timeout=2)
        if cube:
            # Freeze Cozmo for 2 seconds
            #time.sleep(2)
            # Calculate the approximate distance in millimeters
            print(cube.pose.position.x)
            print(cube.pose.position.y)
            distance_to_cube_mm = cube.pose.position.x ** 2 + cube.pose.position.y ** 2
            distance_to_cube_mm = distance_to_cube_mm ** 0.5  # Pythagorean theorem

            # Convert distance from mm to inches
            distance_to_cube_inches = distance_to_cube_mm / 25.4 / 3
            print("front distance = "+str(distance_to_cube_inches))

            # Round the distance to the nearest whole number
            rounded_distance = round(distance_to_cube_inches, 0)
            print(rounded_distance)
            return rounded_distance - 1
        else:
            return None
    except asyncio.TimeoutError:
        print("Cube detection timed out.")
        return None



def look_around_and_find_qr_code(robot):
    # Move Cozmo's head to look around
    robot.set_head_angle(degrees(0)).wait_for_completed()  # Look straight ahead
    robot.set_head_angle(degrees(-25)).wait_for_completed()  # Look down slightly

    # Enable QR code detection
    robot.world.add_event_handler(cozmo.objects.EvtObjectAppeared, on_qr_code_detected)

    # Perform a scan (rotate in place)
    robot.turn_in_place(degrees(360)).wait_for_completed()

    # Disable QR code detection
    robot.world.remove_event_handler(cozmo.objects.EvtObjectAppeared, on_qr_code_detected)




class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(80 * 60 * 32, 128)  # Updated input size

        # Output layers
        self.fc_classification = nn.Linear(128, 1)  # For binary classification
        self.fc_regression = nn.Linear(128, 1)  # For regression (left distance)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        # Classification output (is_there_an_obstacle)
        out_classification = torch.sigmoid(self.fc_classification(x))

        # Regression output (left_distance)
        out_regression = self.fc_regression(x)

        return out_classification, out_regression





transform = transforms.Compose([
    transforms.Resize((320, 240)),
    transforms.ToTensor(),
])

model = CustomCNN()

model.load_state_dict(torch.load("./ml_models/cozmo_model2.pt"))
model.eval()
# Add a batch dimension to the image

def manual_obstacle_update(mymap):
    while True:
        user_input = input("Update obstacles? (y/n): ")
        if user_input.lower() == 'n':
            return None
        elif user_input.lower() == 'y':
            delete_x, delete_y = map(int, input("Enter X,Y to delete obstacle: ").split(','))
            mymap.delete_obstacle(delete_x, delete_y)
            add_x, add_y = map(int, input("Enter X,Y to add obstacle: ").split(','))
            mymap.add_obstacle(add_x, add_y)
            path = mymap.a_star_search()
            mymap.display_map(path)
            return path
        else:
            return None


def calculate_obstacle_position(current_node, orientation, front_dis, side):
    print(current_node, orientation, front_dis, side)
    # Convert front distance and side info to grid coordinates
    x, y = current_node

    if orientation == 0:
        print("1")
        obstacle_y = y + (1 if side == 1 else -1 if side == 2 else 0)
        obstacle_x = x +  front_dis
    elif orientation == 90:
        print("2")
        obstacle_y = y - front_dis
        obstacle_x = x - (1 if side == 1 else -1 if side == 2 else 0)
    elif orientation == 180:
        print("3")
        obstacle_y = y -  (1 if side == 2 else -1 if side == 1 else 0)
        obstacle_x = x - (front_dis)
    elif orientation == -90 or orientation == 270:  # Facing left
        print("4")
        obstacle_x = x + front_dis
        obstacle_y = y + (1 if side == 1 else -1 if side == 2 else 0)
    else:
        # Default case, might indicate an error or unexpected orientation value
        print(f"Unexpected orientation value: {orientation}")
        return None, None

    return int(obstacle_x), int(obstacle_y)


def turn(robot, current_orientation, desired_orientation):
    # Determine the angle to turn from the current orientation to the desired orientation
    angle_to_turn = desired_orientation - current_orientation
    robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()
    return desired_orientation  # Update the current orientation

def follow_path_simple(robot: cozmo.robot.Robot, mymap):
    robot.camera.color_image_enabled = True
    robot.camera.image_stream_enabled = True
    robot.set_lift_height(1).wait_for_completed()
    robot.set_head_angle(cozmo.util.degrees(0 )).wait_for_completed()

    unit_inches = 3  # Each move Cozmo makes is 3 inches
    current_orientation = 0  # Assuming Cozmo starts facing upwards (0 degrees)
    path = mymap.a_star_search()
    step = 0


    while True:
        current_node = path[0]
        next_node = path[1]
        dy, dx = next_node[0] - current_node[0], next_node[1] - current_node[1]


        if dx == 1:
            desired_orientation = -90
        elif dx == -1:
            desired_orientation = 90
        elif dy == 1:
            desired_orientation = 0
        elif dy == -1:
            desired_orientation = 180

        if dx != 0 or dy != 0:
            current_orientation = turn(robot, current_orientation, desired_orientation)
            robot.drive_straight(distance_inches(unit_inches), speed_mmps(50)).wait_for_completed()
            step  += 1
            mymap.set_current_point(next_node[1], next_node[0])
            latest_image = robot.world.latest_image
            if latest_image is not None:
                image = latest_image.raw_image
                image.show()
                # image = Image.open(img_path).convert('RGB')

                image = image.convert('RGB')
                image = transform(image).unsqueeze(0)


                output, side = model(image)
                #print((output[0][0]))
                #print(torch.round(side[0][0]))
                if (output[0][0] > 0.9 and (int(torch.round(side[0][0]).item() <= 1))):  # there is an obs

                    front_dis = measure_distance_to_cube(robot)
                    # Continue even if front_dis is None or if the obstacle is within a certain threshold
                    if front_dis is not None and front_dis < 5:
                        # Check if the obstacle is within a certain threshold
                        obstacle_x, obstacle_y = calculate_obstacle_position(next_node, current_orientation,
                                                                          front_dis, 0)
                        print(f"Obstacle at: ({obstacle_x}, {obstacle_y})")
                        mymap.add_obstacle(obstacle_x, obstacle_y)


            path = mymap.a_star_search()
            mymap.display_map(path)
            """
            put code here 
            """

            #path2 = manual_obstacle_update(mymap)
            #if(path2 != None):
                #path = path2
            if path == None:
                robot.say_text(" path been blocked ").wait_for_completed()
                return
            path = path[0:]
            if(len(path) == 1):
                robot.say_text("I have reached the destination!").wait_for_completed()
                print("Destination reached")
                look_around_and_find_qr_code(robot)
                return


cozmo.run_program(lambda robot: follow_path_simple(robot, mymap))
