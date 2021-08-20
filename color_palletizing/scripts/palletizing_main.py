#!/usr/bin/env python3
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from jetmax_control.msg import SetServo
import hiwonder
import apriltag
import yaml

ROS_NODE_NAME = "palletizing"
IMAGE_PROC_SIZE = 640, 480
TAG_SIZE = 33.30
TARGET_POSITIONS = (((232, -75, 95), -18), ((232, -75, 135), -18), ((232, -75, 185), -18))
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()
at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))


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


class Palletizing:
    def __init__(self):
        self.is_running = False
        self.moving_block = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.level = 0
        self.lock = threading.RLock()
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None

    def reset(self):
        self.is_running = False
        self.moving_block = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.level = 0

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/block_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)


def rotation_mtx_to_euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def moving():
    try:
        tag = state.moving_block
        params = np.array([state.K[0][0], state.K[1][1], state.K[0][2], state.K[1][2]])
        pose_mtx, a, b = at_detector.detection_pose(tag, camera_params=params, tag_size=TAG_SIZE)
        angle = rotation_mtx_to_euler(pose_mtx)[2] * (180 / math.pi)

        cur_x, cur_y, cur_z = jetmax.position
        rect_x, rect_y = state.moving_block.center
        p = np.asarray([rect_x, rect_y]).reshape((1, 1, 2))
        w = camera_to_world(state.K, state.R, state.T, p)[0][0]
        x, y, _ = w
        print(w)
        if angle < -45:  # ccw -45 ~ -90
            angle = -(-90 - angle)
        if angle > 45:
            angle = -(90 - angle)

        new_x, new_y = cur_x + x, cur_y + y
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass

        angle = angle + -arm_angle

        dist = math.sqrt(x * x + y * y + 120 * 120)
        t = dist / 140
        hiwonder.pwm_servo1.set_position(90 + angle, 0.1)
        jetmax.set_position((new_x, new_y, 120), t)
        rospy.sleep(t + 0.1)

        sucker.set_state(True)
        jetmax.set_position((new_x, new_y, 85), 1)
        rospy.sleep(1)

        cur_x, cur_y, cur_z = jetmax.position
        (x, y, z), angle = TARGET_POSITIONS[state.level]
        jetmax.set_position((cur_x, cur_y, 200), 0.8)
        rospy.sleep(0.8)

        state.level += 1
        if state.level == 3:
            state.level = 0

        hiwonder.pwm_servo1.set_position(90 + angle, 0.5)
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 120
        jetmax.set_position((x, y, 200), t)
        rospy.sleep(t + 0.1)

        jetmax.set_position((x, y, z), 1)
        rospy.sleep(1)

        sucker.release(3)
        jetmax.set_position((x, y, 202), 1)
        rospy.sleep(0.5)

    finally:
        # Go home
        sucker.release(3)
        hiwonder.pwm_servo1.set_position(90, 0.5)
        jetmax.go_home(1.5)
        rospy.sleep(1.5)
        with state.lock:
            state.moving_block = None
            state.runner = None


def image_proc(img):
    if state.runner is not None:
        return img
    frame_gray = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2GRAY)
    tags = at_detector.detect(frame_gray)
    for tag in tags:
        corners = tag.corners.reshape(1, -1, 2).astype(int)
        center = tag.center.astype(int)
        cv2.drawContours(img, corners, -1, (255, 0, 0), 3)
        cv2.circle(img, tuple(center), 5, (255, 255, 0), 10)
        cv2.putText(img, "id:%d" % tag.tag_id,
                    (center[0], center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(tags) > 0:
        if state.moving_block is None:
            state.moving_block = tags[0]
        else:
            new_tag = tags[0]
            if new_tag.tag_id != state.moving_block.tag_id:
                state.count = 0
            else:
                state.count += 1
                if state.count > 10:
                    state.count = 0
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.runner.start()
            state.moving_block = tags[0]
    else:
        state.count = 0
        if state.moving_block is not None:
            state.moving_block = None
    img_h, img_w = img.shape[:2]
    cv2.line(img, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
    cv2.line(img, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)
    return img


def image_callback(ros_image):
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    frame_result = image.copy()
    with state.lock:
        if state.is_running:
            frame_result = image_proc(frame_result)
    rgb_image = frame_result.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def enter_func(msg):
    rospy.loginfo("Enter color palletizing")
    exit_func(msg)
    jetmax.go_home()
    state.reset()
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    return TriggerResponse(success=True)


def exit_func(msg):
    rospy.loginfo("Exit color palletizing")
    state.is_running = False
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
        state.heartbeat_timer = None
    if isinstance(state.runner, threading.Thread):
        state.runner.join()
    if isinstance(state.image_sub, rospy.Subscriber):
        state.image_sub.unregister()
        state.image_sub = None
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    return TriggerResponse(success=True)


def set_running(msg: SetBoolRequest):
    if msg.data:
        rospy.loginfo("Start running color palletizing")
        state.is_running = True
    else:
        rospy.loginfo("stop running object tracking")
        state.is_running = False
        state.level = 0
    return SetBoolResponse(success=True)


def heartbeat_timeout_cb():
    rospy.loginfo("Heartbeat timeout. exiting...")
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
    state = Palletizing()
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    state.load_camera_params()
    if state.camera_params is None:
        rospy.logerr('Can not load camera parameters')
        sys.exit(-1)
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # Register result image pub
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        sys.exit(0)
