#!/usr/bin/env python3
import sys
import cv2
import math
import threading
import numpy as np
import rospy
from sensor_msgs.msg import Image
from color_sorting.srv import SetTarget, SetTargetResponse, SetTargetRequest
from jetmax_control.msg import SetJetMax
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
import hiwonder
from hiwonder import serial_servo as ssr
import yaml
import os

ROS_NODE_NAME = "camera_cal"
IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = 270, 280.6


class ColorSortingCalState:
    def __init__(self):
        self.lock = threading.RLock()
        self.heartbeat_timer = None
        self.position = 0, 0, 0
        self.image_sub = None
        self.count = 0
        self.color = {'min': (0, 0, 0), 'max': (255, 255, 255)}
        self.center = 320, 240
        self.rect = 0, 0, 640, 480
        self.is_running = False

    def reset(self):
        self.count = 0
        self.center = 320, 240
        self.rect = 0, 0, 640, 480
        self.is_running = False


state = ColorSortingCalState()


def get_area_max_contour(contours, threshold):
    # 找出面积最大的轮廓
    # 参数为要比较的轮廓的列表
    contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)
    contour_area = list(filter(lambda c: c[1] > threshold, contour_area))
    if len(contour_area) > 0:
        return max(contour_area, key=lambda c: c[1])
    return None


def init():
    hiwonder.motor1.set_speed(0)
    hiwonder.motor2.set_speed(100)
    hiwonder.pwm_servo1.set_position(90, 1000)
    state.reset()
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    hiwonder.motor2.set_speed(0)


def image_proc(img):
    img_h, img_w = img.shape[:2]
    frame_gb = cv2.GaussianBlur(img, (3, 3), 5)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_RGB2LAB)  # 将图像转换到LAB空间

    frame_mask = cv2.inRange(frame_lab, tuple(state.color['min']), tuple(state.color['max']))  # 对原图像和掩模进行位运算
    eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 腐蚀
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出轮廓
    max_contour_area = get_area_max_contour(contours, 200)

    if max_contour_area is not None:  # 画面中找到了选中的颜色
        contour, area = max_contour_area
        rect = cv2.minAreaRect(contour)
        center_x, center_y = rect[0]
        cv2.circle(img, (int(center_x), int(center_y)), 1, (255, 255, 255), 35)
        rect = list(rect)
        l_c_x, l_c_y = state.center
        n_x, n_y = l_c_x * 0.98 + center_x * 0.02, l_c_y * 0.98 + center_y * 0.02
        cv2.circle(img, (int(n_x), int(n_y)), 1, (255, 0, 0), 30)
        cv2.line(img, (int(n_x), 0), (int(n_x), 479), (255, 255, 0), 2)
        cv2.line(img, (0, int(n_y)), (639, int(n_y)), (255, 255, 0), 2)
        state.center = (n_x, n_y)

        x, y, w, h = cv2.boundingRect(contour)
        l_x, l_y, l_w, l_h = state.rect
        n_x = l_x * 0.98 + x * 0.02
        n_y = l_y * 0.98 + y * 0.02
        n_w = l_w * 0.98 + w * 0.02
        n_h = l_h * 0.98 + h * 0.02
        cv2.rectangle(img, (int(n_x), int(n_y)), (int(n_x + n_w), int(n_y + n_h)), (255, 255, 0), 2)
        state.rect = (n_x, n_y, n_w, n_h)
    else:
        state.rect = (0, 0, 640, 480)
        state.center = (320, 240)
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
    rospy.loginfo("enter object tracking")
    exit_func(msg)
    init()
    rospy.ServiceProxy('/usb_cam/start_capture', Empty)()
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    return TriggerResponse(success=True)


def exit_func(msg):
    rospy.loginfo("exit object tracking")
    state.is_running = False
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    if isinstance(state.image_sub, rospy.Subscriber):  # 注销图像节点的订阅, 这样回调就不会被调用了
        rospy.loginfo('unregister image')
        state.image_sub.unregister()
        state.image_sub = None
    return TriggerResponse(success=True)


def set_running(msg: SetBoolRequest):
    rospy.loginfo('set running' + str(msg))
    with state.lock:
        state.is_running = msg.data
        color_ranges_ = rospy.get_param('/lab_config_manager/color_range_list', {})
        state.color = color_ranges_['blue']
    return [True, '']


def heartbeat_timeout_cb():
    rospy.loginfo('heartbeat timeout. exiting')
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


def save_cb(msg: TriggerRequest):
    global block_params, card_params
    rospy.loginfo(("save", msg))
    with state.lock:
        _, _, w, h = state.rect
        w_p = w / 40.0 * 100.0
        h_p = h / 40.0 * 100.0
        c_x, c_y = state.center
        card_params = [c_x, c_y, w_p, h_p]
        rospy.set_param('~card', card_params)
        s = yaml.dump({
            'block': block_params,
            'card': card_params}, default_flow_style=False)
        with open(os.path.join(sys.path[0], '../config/camera_cal.yaml'), 'w') as f:
            f.write(s)
    return [True, '']


def up_cb(msg: TriggerRequest):
    pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    pub.publish(SetJetMax(
        x=hiwonder.JetMax.ORIGIN[0],
        y=hiwonder.JetMax.ORIGIN[1],
        z=hiwonder.JetMax.ORIGIN[2],
        duration=2
    ))
    return [True, '']


def down_cb(msg):
    pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    pub.publish(SetJetMax(
        x=hiwonder.JetMax.ORIGIN[0],
        y=hiwonder.JetMax.ORIGIN[1],
        z=hiwonder.JetMax.ORIGIN[2] - 155,
        duration=2
    ))
    return [True, '']


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    rospy.sleep(0.2)
    init()
    color_ranges = rospy.get_param('/lab_config_manager/color_range_list', {})
    state.color = color_ranges['blue']

    card_params = rospy.get_param('~card', [320, 240, 260, 290])
    block_params = rospy.get_param('~block', [320, 240, 260, 290])

    rospy.set_param('~/card', card_params)
    rospy.set_param('~/block', block_params)

    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # register image publisher
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running)
    up_srv = rospy.Service('/%s/up' % ROS_NODE_NAME, Trigger, up_cb)
    down_srv = rospy.Service('/%s/down' % ROS_NODE_NAME, Trigger, down_cb)
    save_srv = rospy.Service('/%s/save' % ROS_NODE_NAME, Trigger, save_cb)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        sys.exit(0)
