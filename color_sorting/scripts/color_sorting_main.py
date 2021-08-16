#!/usr/bin/env python3
import sys
import cv2
import math
import threading
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from color_sorting.srv import SetTarget, SetTargetResponse, SetTargetRequest
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
import hiwonder

ROS_NODE_NAME = "color_sorting"
IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = 260, 280
SUCKER_PIXEL_X, SUCKER_PIXEL_Y = 328.3, 380.23
IMAGE_PROC_SIZE = 640, 480

jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()


class ColorSortingState:
    def __init__(self):
        self.target_colors = {}
        self.target_positions = {
            'green': ((218, 20, 90), 10),
            'red': ((218, -25, 90), -5),
            'blue': ((218, 65, 90), 18)
        }
        self.heartbeat_timer = None
        self.is_running = False
        self.is_moving = False
        self.image_sub = None
        self.moving_color = None
        self.lock = threading.RLock()
        self.runner = None
        self.count = 0

    def reset(self):
        self.target_colors = {}
        self.is_running = False
        self.is_moving = False
        self.moving_color = None
        self.count = 0


state = ColorSortingState()


def moving():
    rect, box, color_name = state.moving_color
    cur_x, cur_y, cur_z = jetmax.position
    rect_x, rect_y = rect[0]

    try:
        x = (SUCKER_PIXEL_X - rect_x) * (100.0 / IMAGE_PIXEL_PRE_100MM_X)
        y = (SUCKER_PIXEL_Y - rect_y) * (100.0 / IMAGE_PIXEL_PRE_100MM_Y)

        # Calculate the distance between the current position and the target position to control the movement speed
        dist = math.sqrt(x * x + y * y + 120 * 120)
        t = dist / 160

        angle = rect[2]
        if angle < -45:  # ccw -45 ~ -90
            angle = -(-90 - angle)

        new_x, new_y = cur_x + x, cur_y - y - 30
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass

        print(angle, arm_angle, arm_angle + angle)
        angle =  angle + -arm_angle

        # Pick up the block
        hiwonder.pwm_servo1.set_position(90 + angle, 0.1)
        jetmax.set_position((new_x, new_y, 120), t)
        rospy.sleep(t)
        sucker.set_state(True)  # Turn on the air pump
        jetmax.set_position((new_x, new_y, 80), 1)
        rospy.sleep(1)

        cur_x, cur_y, cur_z = jetmax.position
        jetmax.set_position((cur_x, cur_y, 180), 0.8)
        rospy.sleep(0.8)
        hiwonder.pwm_servo1.set_position(90, 0.1)
        #jetmax.go_home(1)
        #rospy.sleep(5)

        # Go to the target position
        (x, y, z), angle = state.target_positions[color_name]
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt(((cur_x - x) ** 2 + (cur_y - y) ** 2)) / 180.0
        hiwonder.pwm_servo1.set_position(90 + angle, 0.5)
        jetmax.set_position((x, y, 140), t)
        rospy.sleep(t)
        jetmax.set_position((x, y, z), 1)
        rospy.sleep(1)

        # Put down the block
        sucker.set_state(False)  # Turn off the air pump
        jetmax.set_position((x, y, 140), 0.8)
        rospy.sleep(0.8)
    finally:
        # Go home
        sucker.set_state(False)
        hiwonder.pwm_servo1.set_position(90, 0.5)
        jetmax.go_home(2)
        rospy.sleep(2.5)
        state.runner = None
        state.is_moving = False


def image_proc(img):
    img_h, img_w = img.shape[:2]
    org_img = np.copy(img)
    frame_gb = cv2.GaussianBlur(img, (5, 5), 5)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_RGB2LAB)  # Convert rgb to lab

    blocks = []
    for color_name, color in state.target_colors.items():  # Loop through all selected colors
        frame_mask = cv2.inRange(frame_lab, tuple(color['min']), tuple(color['max']))
        eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)
        contour_area = list(filter(lambda c: c[1] > 1000, contour_area))  # Eliminate contours that are too small

        if len(contour_area) > 0:
            for contour, area in contour_area:  # Loop through all the contours found
                rect = cv2.minAreaRect(contour)
                center_x, center_y = rect[0]
                box = np.int0(cv2.boxPoints(rect))  # The four vertices of the minimum-area-rectangle
                # If the contour is out of the recognition zone, skip it
                if center_y > (SUCKER_PIXEL_Y + IMAGE_PIXEL_PRE_100MM_Y / 100 * 80):
                    continue
                for i, p in enumerate(box):
                    box[i, 0] = int(hiwonder.misc.val_map(p[0], 0, IMAGE_PROC_SIZE[0], 0, img_w))
                    box[i, 1] = int(hiwonder.misc.val_map(p[1], 0, IMAGE_PROC_SIZE[1], 0, img_h))
                cv2.drawContours(org_img, [box], -1, hiwonder.COLORS[color_name.upper()], 2)
                cv2.circle(org_img, (int(center_x), int(center_y)), 1, hiwonder.COLORS[color_name.upper()], 5)
                rect = list(rect)
                rect[0] = (center_x, center_y)
                blocks.append((rect, box, color_name))

    if len(blocks) > 0:
        if state.moving_color is None:
            # Choose the contour with the largest area as the next target
            state.moving_color = max(blocks, key=lambda tmp: tmp[0][1][0] * tmp[0][1][1])
        else:
            # Find the rectangle with the smallest distance from the last rectangle and update the data
            rect, _, _ = state.moving_color
            moving_x, moving_y = rect[0]
            blocks = list(map(lambda tmp: (tmp, math.sqrt((moving_x - tmp[0][0][0]) ** 2 +
                                                          (moving_y - tmp[0][0][1]) ** 2)), blocks))
            blocks.sort(key=lambda tmp: tmp[1])
            moving_color, _ = blocks[0]
            x, y = moving_color[0][0]
            cv2.drawContours(org_img, [moving_color[1]], -1, (255, 255, 255), 2)
            cv2.circle(org_img, (int(x), int(y)), 1, (255, 255, 255), 5)

            if blocks[0][1] < 30:
                state.count += 1
                if state.count > 5:
                    rect, box, color_name = moving_color
                    (x, y), (w, h), angle = rect
                    (o_x, o_y), _, o_angle = state.moving_color[0]
                    o_x = x * 0.2 + o_x * 0.8
                    o_y = y * 0.2 + o_y * 0.8
                    o_angle = angle * 0.2 + o_angle * 0.8
                    rect = (o_x, o_y), (w, h), o_angle
                    moving_color = rect, box, color_name
                    print(o_angle)
                    if state.count > 30 and not state.is_moving:
                        state.moving_color = moving_color
                        state.is_moving = True
                        state.count = 0
                        state.runner = threading.Thread(target=moving, daemon=True)  # Move block
                        state.runner.start()
            else:
                state.count = 0
            if not state.is_moving:
                state.moving_color = moving_color
    else:
        if state.moving_color is not None and not state.is_moving:
            state.moving_color = None
        state.count = 0

    cv2.line(org_img, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
    cv2.line(org_img, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)
    return org_img


def image_callback(ros_image):
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    frame_result = image
    with state.lock:
        if state.is_running:
            frame_result = image_proc(frame_result)
    rgb_image = frame_result.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def enter_func(msg):
    rospy.loginfo("Enter color sorting")
    global SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y
    exit_func(msg)
    jetmax.go_home()
    state.reset()
    rospy.ServiceProxy('/usb_cam/start_capture', Empty)()
    SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/block', [SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y])
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    return TriggerResponse(success=True)


def exit_func(msg):
    rospy.loginfo("Exit color sorting")
    state.is_running = False
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    if isinstance(state.runner, threading.Thread):  # If the arm is moving, wait for it to complete
        state.runner.join()
    if isinstance(state.image_sub, rospy.Subscriber):
        state.image_sub.unregister()
        state.image_sub = None
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    return TriggerResponse(success=True)


def set_running(msg: SetBoolRequest):
    rospy.loginfo('Set running' + str(msg))
    if msg.data:
        state.is_running = True
    else:
        state.is_running = False
    return SetBoolResponse()


def set_target_cb(msg: SetTargetRequest):
    global SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y
    try:
        if msg.is_enable:
            color_ranges = rospy.get_param('/lab_config_manager/color_range_list', {})
            with state.lock:
                state.target_colors[msg.color_name] = color_ranges[msg.color_name]
                SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
                    '/camera_cal/block',
                    [SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y])
            rospy.logdebug('set target color ' + str(msg))
        else:
            with state.lock:
                del (state.target_colors[msg.color_name])
            rospy.loginfo('disable target color: ' + msg.color_name)
    except Exception as e:
        rospy.logerr(e)
        return SetTargetResponse(success=False, message=str(e))
    return SetTargetResponse(success=True)


def heartbeat_timeout_cb():
    rospy.loginfo('Heartbeat timeout. exiting')
    rospy.ServiceProxy('/%s/exit' % ROS_NODE_NAME, Trigger)()


def heartbeat_srv_cb(msg: SetBoolRequest):
    """
    Heartbeat callback. A timer will be set, and the exit service will be called when the time is reached

    """
    rospy.logdebug("Heartbeat")
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    if msg.data:
        state.heartbeat_timer = threading.Timer(5, heartbeat_timeout_cb)
        state.heartbeat_timer.start()
    else:
        if isinstance(state.heartbeat_timer, threading.Timer):
            state.heartbeat_timer.cancel()
    return SetBoolResponse(success=msg.data)


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    rospy.sleep(0.2)
    state.reset()
    SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/block',
        [SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y])
    print(SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y)
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # register image publisher
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running)
    set_target_srv = rospy.Service('/%s/set_target' % ROS_NODE_NAME, SetTarget, set_target_cb)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        sys.exit(0)
