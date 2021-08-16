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

ROS_NODE_NAME = "color_palletizing"
IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = 256, 280.5
SUCKER_PIXEL_X, SUCKER_PIXEL_Y = 386.5, 298
IMAGE_PROC_SIZE = 640, 480
TARGET_POSITIONS = (((215, -72, 90), -15), ((215, -72, 130), -15), ((215, -72, 170), -15))
TARGET_COLORS = ('red', 'green', 'blue')
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()


class ColorPalletizing:
    def __init__(self):
        self.is_running = False
        self.moving_color = None
        self.image_sub = None

        self.target_colors = {}
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.step = 0
        self.timestamp = time.time()
        self.lock = threading.RLock()
        self.level = 0

    def reset(self):
        self.is_running = False
        self.moving_color = None
        self.image_sub = None
        self.target_colors = {}
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.level = 0
        self.timestamp = time.time()


state = ColorPalletizing()


def moving():
    try:
        # 计算机械臂当前位置与目标位置的x, y 轴距离
        rect, box, color_name = state.moving_color
        cur_x, cur_y, cur_z = jetmax.position
        rect_x, rect_y = rect[0]

        x = (SUCKER_PIXEL_X - rect_x) * (100.0 / IMAGE_PIXEL_PRE_100MM_X)
        y = (SUCKER_PIXEL_Y - rect_y) * (100.0 / IMAGE_PIXEL_PRE_100MM_Y)

        # 计算气泵旋转舵机要旋转的角度
        # 机械臂吸取物品时的相对相对于初始位置时机械比正前方夹角加物品的旋转角度
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
        angle = angle + -arm_angle 

        # 计算当前位置与目标位置的欧氏距离，以控制运动速度
        dist = math.sqrt(x * x + y * y + 120 * 120)
        t = dist / 160
        hiwonder.pwm_servo1.set_position(90 + angle, 0.1)
        jetmax.set_position((new_x, new_y, 120), t)
        rospy.sleep(t + 0.1)

        sucker.set_state(True)
        jetmax.set_position((new_x, new_y, 80), 0.5)  # 下移机械臂，吸取物品
        rospy.sleep(0.5)

        cur_x, cur_y, cur_z = jetmax.position  # 当前的机械臂位置
        (x, y, z), angle = TARGET_POSITIONS[state.level]  # 获取码垛的位置
        jetmax.set_position((cur_x, cur_y, 200), 0.8)
        rospy.sleep(0.8)

        state.level += 1
        if state.level == 3:
            state.level = 0

        # 机械臂运动到码垛位置上空
        hiwonder.pwm_servo1.set_position(90 + angle, 0.5)
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 120
        jetmax.set_position((x, y, 200), t)
        rospy.sleep(t + 0.1)

        # 机械臂下移到目标码垛位置
        jetmax.set_position((x, y, z), 1)
        rospy.sleep(1)

        sucker.release(3)
        jetmax.set_position((x, y, 202), 1)  # 上提机械臂
        rospy.sleep(0.5)

    finally:
        # Go home
        sucker.release(3)
        hiwonder.pwm_servo1.set_position(90, 0.5)
        jetmax.go_home(1.5)
        rospy.sleep(1.5)
        with state.lock:
            state.moving_color = None
            state.runner = None


def image_proc(img):
    if state.runner is not None:
        return img

    img_h, img_w = img.shape[:2]
    frame_resize = cv2.resize(img, IMAGE_PROC_SIZE, interpolation=cv2.INTER_NEAREST)
    frame_lab = cv2.cvtColor(frame_resize, cv2.COLOR_RGB2LAB)
    blocks = []
    for color_name, color in state.target_colors.items():
        frame_gb = cv2.GaussianBlur(frame_lab, (5, 5), 5)
        frame_mask = cv2.inRange(frame_gb, tuple(color['min']), tuple(color['max']))
        eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)
        contour_area = list(filter(lambda c: c[1] > 1000, contour_area))

        if len(contour_area) > 0:
            for contour, area in contour_area:
                rect = cv2.minAreaRect(contour)
                center_x, center_y = rect[0]
                center_x = int(hiwonder.misc.val_map(center_x, 0, IMAGE_PROC_SIZE[0], 0, img_w))
                center_y = int(hiwonder.misc.val_map(center_y, 0, IMAGE_PROC_SIZE[1], 0, img_h))
                if center_y > SUCKER_PIXEL_Y + (IMAGE_PIXEL_PRE_100MM_Y / 100.0 * 80):
                    continue
                box = np.int0(cv2.boxPoints(rect))
                for i, p in enumerate(box):
                    box[i, 0] = int(hiwonder.misc.val_map(p[0], 0, IMAGE_PROC_SIZE[0], 0, img_w))
                    box[i, 1] = int(hiwonder.misc.val_map(p[1], 0, IMAGE_PROC_SIZE[1], 0, img_h))
                cv2.drawContours(img, [box], -1, hiwonder.COLORS_BGR[color_name.upper()], 2)
                cv2.circle(img, (center_x, center_y), 1, hiwonder.COLORS_BGR[color_name.upper()], 5)
                rect = list(rect)
                rect[0] = (center_x, center_y)
                blocks.append((rect, box, color_name))

    if len(blocks) > 0:
        if state.moving_color is None:
            state.moving_color = max(blocks, key=lambda tmp: tmp[0][1][0] * tmp[0][1][1])  # 状态中没有记下色块的话,记下识别到的面积最大的色块
        else:
            rect, _, _ = state.moving_color
            moving_x, moving_y = rect[0]
            blocks = list(map(lambda tmp: (tmp, math.sqrt((moving_x - tmp[0][0][0]) ** 2 +
                                                          (moving_y - tmp[0][0][1]) ** 2)), blocks))
            blocks.sort(key=lambda tmp: tmp[1])  # 找出与记下的色块中心距离最近的色块
            new_block, dist = blocks[0]
            cv2.drawContours(img, [new_block[1]], -1, (255, 255, 255), 2)
            cv2.circle(img, new_block[0][0], 1, (255, 255, 255), 5)
            if dist < 30:  # 距离最近的色块距离小于设定值就是同一个色块
                state.count += 1
                if state.count > 3:
                    rect, box, color_name = new_block
                    (x, y), (w, h), angle = rect
                    (o_x, o_y), _, o_angle = state.moving_color[0]
                    o_x = x * 0.2 + o_x * 0.8
                    o_y = y * 0.2 + o_y * 0.8
                    o_angle = angle * 0.2 + o_angle * 0.8
                    rect = (o_x, o_y), (w, h), o_angle
                    new_block = rect, box, color_name
                    state.moving_color = new_block
                    if state.count > 30:  # 连续超过30次识别认为稳定识别到
                        state.count = 0
                        state.runner = threading.Thread(target=moving, daemon=True)  # 启动一个线程完成码垛动作
                        state.runner.start()
            else:
                state.count = 0
            state.moving_color = new_block

    else:
        state.count = 0
        if state.moving_color is not None:
            state.moving_color = None

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
    global SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y
    rospy.loginfo("Enter color palletizing")
    exit_func(msg)
    jetmax.go_home()
    state.reset()
    SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/block',
        [SUCKER_PIXEL_X,
         SUCKER_PIXEL_Y,
         IMAGE_PIXEL_PRE_100MM_X,
         IMAGE_PIXEL_PRE_100MM_Y])
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    rospy.ServiceProxy('/usb_cam/start_capture', Empty)()
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
        color_ranges = rospy.get_param('/lab_config_manager/color_range_list', {})
        for color in TARGET_COLORS:
            if color in color_ranges:
                state.target_colors[color] = color_ranges[color]
        state.is_running = True
    else:
        rospy.loginfo("stop running object tracking")
        state.is_running = False
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
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/block',
        [SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y])
    rospy.loginfo((SUCKER_PIXEL_X, SUCKER_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y))
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # Register result image pub
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        sys.exit(0)
