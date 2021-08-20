#!/usr/bin/env python3
import sys
import cv2
import math
import threading
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from color_sorting.srv import SetTarget, SetTargetResponse, SetTargetRequest
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
import hiwonder

ROS_NODE_NAME = "color_sorting"
IMAGE_PROC_SIZE = 640, 480
unit_block_corners = np.asarray([[0, 0, 0],
                                 [20, -20, 0],  # TAG_SIZE = 33.30mm
                                 [-20, -20, 0],
                                 [-20, 20, 0],
                                 [20, 20, 0]],
                                dtype=np.float64)
unit_block_img_pts = None


def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invR
        worldPtPlane = np.dot(inv_r, worldPtCam)  # 3*3 dot 3*1 = 3*1
        # zc
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


class ColorSortingState:
    def __init__(self):
        self.target_colors = {}
        self.target_positions = {
            'red': ((232, -25, 95), -5),
            'green': ((233, 25, 95), 8),
            'blue': ((237, 75, 95), 18)
        }
        self.heartbeat_timer = None
        self.is_running = False
        self.image_sub = None
        self.moving_color = None
        self.lock = threading.RLock()
        self.runner = None
        self.count = 0
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None
        self.WIDTH = None

    def reset(self):
        self.target_colors = {}
        self.is_running = False
        self.moving_color = None
        self.count = 0

    def load_camera_params(self):
        global unit_block_img_pts
        self.camera_params = rospy.get_param('/camera_cal/block_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)
            img_pts, jac = cv2.projectPoints(unit_block_corners, self.R, self.T, self.K, None)
            unit_block_img_pts = img_pts.reshape(5, 2)
            l_p1 = unit_block_img_pts[-1]
            l_p2 = unit_block_img_pts[-2]
            self.WIDTH = math.sqrt((l_p1[0] - l_p2[0]) ** 2 + (l_p1[1] - l_p2[1]) ** 2)
            print(unit_block_img_pts)


def moving():
    rect, box, color_name = state.moving_color
    cur_x, cur_y, cur_z = jetmax.position
    try:
        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array(rect[0]).reshape((1, 1, 2)))[0][0]
        # Calculate the distance between the current position and the target position to control the movement speed
        t = math.sqrt(x * x + y * y + 120 * 120) / 140
        angle = rect[2]
        if angle < -45:  # ccw -45 ~ -90
            angle = -(-90 - angle)

        new_x, new_y = cur_x + x, cur_y + y
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass

        angle = angle + -arm_angle

        # Pick up the block
        hiwonder.pwm_servo1.set_position(90 + angle, 0.1)
        jetmax.set_position((new_x, new_y, 120), t)
        rospy.sleep(t)
        sucker.set_state(True)  # Turn on the air pump
        jetmax.set_position((new_x, new_y, 85), 1)
        rospy.sleep(1.05)

        cur_x, cur_y, cur_z = jetmax.position
        jetmax.set_position((cur_x, cur_y, 180), 0.8)
        rospy.sleep(0.8)
        hiwonder.pwm_servo1.set_position(90, 0.1)

        # Go to the target position
        (x, y, z), angle = state.target_positions[color_name]
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt(((cur_x - x) ** 2 + (cur_y - y) ** 2)) / 180.0
        hiwonder.pwm_servo1.set_position(90 + angle, 0.5)
        jetmax.set_position((x, y, 180), t)
        rospy.sleep(t)
        jetmax.set_position((x, y, z), 1)
        rospy.sleep(1)

        # Put down the block
        sucker.release(3)  # Turn off the air pump
        jetmax.set_position((x, y, 140), 0.8)
        rospy.sleep(0.8)
    except Exception as e:
        rospy.logerr("ERROR")
    finally:
        # Go home
        sucker.release(3)
        hiwonder.pwm_servo1.set_position(90, 0.5)
        jetmax.go_home(2)
        rospy.sleep(2.5)
        state.moving_color = None
        state.runner = None


def point_xy(pt_a, pt_b, r):
    x_a, y_a = pt_a
    x_b, y_b = pt_b
    if x_a == x_b:
        return x_a, y_a + (r / abs((y_b - y_a))) * (y_b - y_a)
    k = (y_a - y_b) / (x_a - x_b)
    b = y_a - k * x_a
    A = k ** 2 + 1
    B = 2 * ((b - y_a) * k - x_a)
    C = (b - y_a) ** 2 + x_a ** 2 - r ** 2
    x1 = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    x2 = (-B - math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    y1 = k * x1 + b
    y2 = k * x2 + b
    dist_1 = math.sqrt((x1 - x_b) ** 2 + (y1 - y_b) ** 2)
    dist_2 = math.sqrt((x2 - x_b) ** 2 + (y2 - y_b) ** 2)
    if dist_1 <= dist_2:
        return x1, y1
    else:
        return x2, y2


def image_proc(img):
    if state.runner is not None:
        return img
    img_h, img_w = img.shape[:2]
    frame_gb = cv2.GaussianBlur(np.copy(img), (5, 5), 5)
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
                box = cv2.boxPoints(rect)  # The four vertices of the minimum-area-rectangle
                box_list = box.tolist()
                box = np.int0(box)
                ap = max(box_list, key=lambda p: math.sqrt((p[0] - state.K[0][2]) ** 2 + (p[1] - state.K[1][2]) ** 2))
                index_ap = box_list.index(ap)
                p1 = box_list[index_ap - 1 if index_ap - 1 >= 0 else 3]
                p2 = box_list[index_ap + 1 if index_ap + 1 <= 3 else 0]
                n_p1 = point_xy(ap, p1, state.WIDTH)
                n_p2 = point_xy(ap, p2, state.WIDTH)

                c_x, c_y = None, None
                if n_p1 and n_p2:
                    x_1, y_1 = n_p1
                    x_2, y_2 = n_p2
                    c_x = (x_1 + x_2) / 2
                    c_y = (y_1 + y_2) / 2
                    cv2.circle(img, (int(n_p1[0]), int(n_p1[1])), 2, (255, 255, 0), 10)
                    cv2.circle(img, (int(n_p2[0]), int(n_p2[1])), 2, (255, 255, 0), 10)
                    cv2.circle(img, (int(c_x), int(c_y)), 2, (0, 0, 0), 10)

                cv2.circle(img, (int(ap[0]), int(ap[1])), 2, (0, 255, 255), 10)
                cv2.drawContours(img, [box], -1, hiwonder.COLORS[color_name.upper()], 2)
                cv2.circle(img, (int(center_x), int(center_y)), 1, hiwonder.COLORS[color_name.upper()], 5)
                rect = list(rect)
                if c_x:
                    rect[0] = c_x, c_y
                else:
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
            cv2.drawContours(img, [moving_color[1]], -1, (255, 255, 255), 2)
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 255), 5)

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
                if state.count > 30:
                    state.count = 0
                    state.moving_color = moving_color
                    state.runner = threading.Thread(target=moving, daemon=True)  # Move block
                    state.runner.start()
                state.moving_color = moving_color
    else:
        state.moving_color = None
        state.count = 0

    cv2.line(img, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
    cv2.line(img, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)
    return img


def image_callback(ros_image):
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    frame_result = np.copy(image)
    with state.lock:
        if state.is_running:
            frame_result = image_proc(frame_result)
    rgb_image = frame_result.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def enter_func(msg):
    rospy.loginfo("Enter color sorting")
    exit_func(msg)
    jetmax.go_home()
    state.reset()
    state.load_camera_params()
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
    try:
        if msg.is_enable:
            color_ranges = rospy.get_param('/lab_config_manager/color_range_list', {})
            with state.lock:
                state.target_colors[msg.color_name] = color_ranges[msg.color_name]
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
    state = ColorSortingState()
    state.load_camera_params()
    if state.camera_params is None:
        rospy.logerr("Can not load camera params")
        sys.exit(-1)
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    rospy.sleep(0.2)
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
