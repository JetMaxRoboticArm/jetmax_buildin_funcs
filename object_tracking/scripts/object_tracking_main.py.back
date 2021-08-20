#!/usr/bin/env python3
import sys
import cv2
import math
import rospy
import time
import queue
import threading
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
from object_tracking.srv import SetTarget, SetTargetResponse, SetTargetRequest
import hiwonder
import numpy as np

ROS_NODE_NAME = 'object_tracking'
TARGET_PIXEL_X, TARGET_PIXEL_Y = 320, 240


class ObjectTracking:
    def __init__(self):
        self.heartbeat_timer = None
        self.lock = threading.Lock()
        self.servo_x = 500
        self.servo_y = 500

        self.color_x_pid = hiwonder.PID(0.07, 0.01, 0.0015)
        self.color_y_pid = hiwonder.PID(0.08, 0.008, 0.001)

        self.target_color_range = None
        self.target_color_name = None
        self.last_color_circle = None
        self.lost_target_count = 0

        self.is_running = False
        self.entered = False

        self.fps = 0.0
        self.tic = time.time()

    def reset(self):
        self.heartbeat_timer = None
        self.is_running = False
        self.servo_x = 500
        self.servo_y = 500
        self.last_color_circle = None
        self.lost_target_count = 0
        self.tic = time.time()
        self.target_color_range = None
        self.target_color_name = None
        self.last_color_circle = None


state = ObjectTracking()
jetmax = hiwonder.JetMax()




def color_tracking(image):
    global TARGET_PIXEL_X, TARGET_PIXEL_Y
    org_image = np.copy(image)
    image = cv2.resize(image, (320, 240))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # RGB转LAB空间
    image = cv2.GaussianBlur(image, (5, 5), 5)

    target_color_range = state.target_color_range
    target_color_name = state.target_color_name

    if target_color_range is not None:
        mask = cv2.inRange(image, tuple(target_color_range['min']), tuple(target_color_range['max']))  # 二值化
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 腐蚀
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出轮廓
        contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)  # 计算各个轮廓的面积
        contour_area = list(filter(lambda c: c[1] > 300, contour_area))  # 剔除面积过小的轮廓
        circle = None
        if len(contour_area) > 0:
            if state.last_color_circle is None:
                contour, area = max(contour_area, key=lambda c_a: c_a[1])
                circle = cv2.minEnclosingCircle(contour)
            else:
                (last_x, last_y), last_r = state.last_color_circle
                circles = map(lambda c: cv2.minEnclosingCircle(c[0]), contour_area)
                circle_dist = list(map(lambda c: (c, math.sqrt(((c[0][0] - last_x) ** 2) + ((c[0][1] - last_y) ** 2))),
                                       circles))
                circle, dist = min(circle_dist, key=lambda c: c[1])
                if dist < 50:
                    circle = circle

        if circle is not None:
            # 可以使用极坐标进行定位， 这里不用精确计算坐标，直接控制两个舵机来定位
            state.lost_target_count = 0
            (c_x, c_y), c_r = circle
            c_x = hiwonder.misc.val_map(c_x, 0, 320, 0, 640)
            c_y = hiwonder.misc.val_map(c_y, 0, 240, 0, 480)
            c_r = hiwonder.misc.val_map(c_r, 0, 320, 0, 640)

            x = c_x - TARGET_PIXEL_X
            if abs(x) > 30:
                state.color_x_pid.SetPoint = 0
                state.color_x_pid.update(x)
                state.servo_x += state.color_x_pid.output
            else:
                state.color_x_pid.update(0)

            y = c_y - TARGET_PIXEL_Y
            if abs(y) > 30:
                state.color_y_pid.SetPoint = 0
                state.color_y_pid.update(y)
                state.servo_y -= state.color_y_pid.output
            else:
                state.color_y_pid.update(0)
            if state.servo_y < 350:
                state.servo_y = 350
            if state.servo_y > 650:
                state.servo_y = 650
            jetmax.set_servo(1, int(state.servo_x), duration=0.02)
            jetmax.set_servo(2, int(state.servo_y), duration=0.02)
            color_name = target_color_name.upper()
            org_image = cv2.circle(org_image, (int(c_x), int(c_y)), int(c_r), hiwonder.COLORS[color_name], 3)
            state.last_color_circle = circle
        else:
            state.lost_target_count += 1
            if state.lost_target_count > 15:
                state.lost_target_count = 0
                state.last_color_circle = None
    return org_image


def image_callback(ros_image):
    if not state.entered:
        rospy.sleep(0.1)
        return
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    frame_result = image
    with state.lock:
        if state.is_running:
            frame_result = color_tracking(np.copy(image))
    toc = time.time()
    curr_fps = 1.0 / (state.tic - toc)
    state.fps = curr_fps if state.fps == 0.0 else (state.fps * 0.95 + curr_fps * 0.05)
    state.tic = toc
    rgb_image = frame_result.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def enter_func(msg):
    rospy.loginfo("enter object tracking")
    exit_func(msg)  # 先退出一次, 简化过程
    with state.lock:
        state.reset()
        state.entered = True
    return [True, '']



def exit_func(msg):
    rospy.loginfo("exit object tracking")
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    with state.lock:
        state.entered = False
        state.is_running = False
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    return TriggerResponse(success=True)


def set_running_color_cb(msg: SetBoolRequest):
    rospy.loginfo("start running color tracking")
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    with state.lock:
        state.reset()
        state.is_running = msg.data
        if not msg.data:
            jetmax.go_home()
            rospy.sleep(0.1)
    return [True, '']


def heartbeat_timeout_cb():
    rospy.loginfo("heartbeat timeout. exiting...")
    rospy.ServiceProxy('/%s/exit' % ROS_NODE_NAME, Trigger)()


def heartbeat_srv_cb(msg: SetBoolRequest):
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    rospy.logdebug("Heartbeat")
    if msg.data:
        state.heartbeat_timer = threading.Timer(5, heartbeat_timeout_cb)
        state.heartbeat_timer.start()
    else:
        if isinstance(state.heartbeat_timer, threading.Timer):
            state.heartbeat_timer.cancel()
    return SetBoolResponse(success=msg.data)


def set_target_cb(msg: SetTargetRequest):
    color_ranges = rospy.get_param('/lab_config_manager/color_range_list', {})
    rospy.logdebug(color_ranges)
    with state.lock:
        if msg.color_name in color_ranges:
            state.target_color_name = msg.color_name
            state.target_color_range = color_ranges[msg.color_name]
            return [True, '']
        else:
            state.target_color_name = None
            state.target_color_range = None
            return [False, '']


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    state.reset()
    TARGET_PIXEL_X, TARGET_PIXEL_Y, _, _ = rospy.get_param('/camera_cal/block', (320, 240))
    print(TARGET_PIXEL_X, TARGET_PIXEL_Y)

    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback, queue_size=1)  # 订阅摄像头画面
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=2)  # register result image pub
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    color_running_srv = rospy.Service('/%s/set_running_color' % ROS_NODE_NAME, SetBool, set_running_color_cb)
    set_running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running_color_cb)
    set_target_srv = rospy.Service("/%s/set_target" % ROS_NODE_NAME, SetTarget, set_target_cb)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)
    try:
        rospy.spin()
    except:
        sys.exit(0)
