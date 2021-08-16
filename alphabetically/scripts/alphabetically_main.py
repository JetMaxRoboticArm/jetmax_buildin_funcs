#!/usr/bin/env python3
import os
import sys
import cv2
import math
import time
import queue
import random
import threading
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from jetmax_control.msg import SetServo
import hiwonder
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "alphabetically"

IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = 203.6, 230.5
TARGET_PIXEL_X, TARGET_PIXEL_Y = 400, 255
IMAGE_SIZE = 640, 480

CHARACTERS_ENGINE_PATH = os.path.join(sys.path[0], 'characters_v5_160.trt')
CHARACTER_LABELS = tuple([i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
CHARACTER_NUM = 26

NUMBERS_ENGINE_PATH = os.path.join(sys.path[0], 'number_v5_160.trt')
NUMBERS_LABELS = tuple([i for i in '0123456789+-*/='])
NUMBERS_NUM = 15

TRT_INPUT_SIZE = 160
COLORS = tuple([tuple([random.randint(10, 255) for j in range(3)]) for i in range(CHARACTER_NUM + NUMBERS_NUM)])

GOAL_POSITIONS = (
    (-210, -84, 55, 17), (-210, -46, 55, 7), (-210, -6, 55, -3), (-210, 37, 55, -13),
    (-161, -80, 55, 20), (-161, -46, 55, 10), (-161, -6, 55, -1), (-161, 37, 55, -13),
    (-115, -75, 55, 27), (-115, -38, 55, 15), (-115, -4, 55, 2), (-115, 34, 55, -13),
)


class Alphabetically:
    def __init__(self):
        self.running_mode = 0
        self.moving_box = None
        self.image_sub = None
        self.is_moving = False

        self.heartbeat_timer = None
        self.runner = None
        self.moving_count = 0
        self.count = 0
        self.step = 0
        self.lock = threading.RLock()
        self.fps_t0 = time.time()
        self.fps = 0

    def reset(self):
        self.running_mode = 0
        self.moving_box = None
        self.is_moving = False
        self.moving_count = 0
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.step = 0
        self.fps_t0 = time.time()
        self.fps = 0


print(CHARACTERS_ENGINE_PATH)
yolov5_chars = Yolov5TensorRT(CHARACTERS_ENGINE_PATH, TRT_INPUT_SIZE, CHARACTER_NUM)
yolov5_nums = Yolov5TensorRT(NUMBERS_ENGINE_PATH, TRT_INPUT_SIZE, NUMBERS_NUM)
image_queue = queue.Queue(maxsize=1)

state = Alphabetically()
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()


def moving():
    try:
        # 计算机械臂当前位置与目标位置的x, y 轴距离
        c_x, c_y, cls_id, cls_conf = state.moving_box
        cur_x, cur_y, cur_z = jetmax.position

        x = (TARGET_PIXEL_X - c_x) * (100.0 / IMAGE_PIXEL_PRE_100MM_X)
        y = (TARGET_PIXEL_Y - c_y) * (100.0 / IMAGE_PIXEL_PRE_100MM_Y)
        print("dist", x, y)

        # 计算当前位置与目标位置的欧氏距离，以控制运动速度
        dist = math.sqrt(x * x + y * y + 120 * 120)
        t = dist / 120

        # 计算气泵旋转舵机要旋转的角度
        # 机械臂吸取物品时的相对相对于初始位置时机械比正前方夹角加物品的旋转角度
        new_x, new_y = cur_x + x, cur_y - y
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass
        print('arm_angel', arm_angle)
        # print(angle, arm_angle, angle_c)
        jetmax.set_position((new_x, new_y, 70), t)
        rospy.sleep(t + 0.5)

        sucker.set_state(True)
        jetmax.set_position((new_x, new_y, 45), 0.5)  # 下移机械臂，吸取物品
        rospy.sleep(0.55)

        # 获取码垛的位置
        x, y, z, angle = GOAL_POSITIONS[state.moving_count]

        cur_x, cur_y, cur_z = jetmax.position  # 当前的机械臂位置
        jetmax.set_position((cur_x, cur_y, 100), 0.8)
        rospy.sleep(0.8)

        # 机械臂运动到target position上空
        hiwonder.pwm_servo1.set_position(90 + angle + arm_angle, 0.1)
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 150
        jetmax.set_position((x, y, z + 30), t)
        rospy.sleep(t)

        # 机械臂下移到目标码垛位置
        jetmax.set_position((x, y, z), 0.8)
        rospy.sleep(0.8)

        sucker.release(3)
        jetmax.set_position((x, y, z + 30), 0.8)  # 上提机械臂
        rospy.sleep(0.1)
        hiwonder.pwm_servo1.set_position(90, 0.4)  # 恢复吸盘角度

    finally:
        sucker.release(3)
        # 回到初始位置
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - jetmax.ORIGIN[0]) ** 2 + (cur_y - jetmax.ORIGIN[1]) ** 2) / 120
        jetmax.go_home(t)
        hiwonder.pwm_servo1.set_position(90, 0.2)  # 恢复吸盘角度
        rospy.sleep(t + 0.2)
        with state.lock:
            state.moving_box = None
            state.is_moving = False
            state.moving_count += 1
            if state.moving_count >= len(GOAL_POSITIONS):
                state.moving_count = 0
        print("FINISHED")


def image_proc_chars(img_in):
    outputs = yolov5_chars.detect(img_in)
    boxes, confs, classes = yolov5_chars.post_process(img_in, outputs, 0.90)
    cards = []
    width, height = IMAGE_SIZE
    result_image = img_in
    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        if cls_conf < 0.9:
            continue
        thr = TARGET_PIXEL_Y + (IMAGE_PIXEL_PRE_100MM_Y / 100.0 * 120.0)
        if y2 > thr:  # 如果坐标超出了识别范围(识别状态下吸盘正下发位置的像素坐标+120mm), 可能识别到错误的东西了，放弃掉
            continue
        cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        #rospy.loginfo((TARGET_PIXEL_Y, y2))
        result_image = cv2.putText(img_in, CHARACTER_LABELS[cls_id] + " " + str(float(cls_conf))[:4],
                                   (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls_id], 2)
        result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls_id], 3)

    if state.is_moving:  # 机械臂在运动过程中不再进行新的运动
        return result_image

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = min(cards, key=lambda x: x[4])  # 识别到的所有卡牌中类别id最小的一个
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, cls_id, cls_conf  # 存起来
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            state.count = 0
        else:
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            cards = [((x1 + x2) / 2,
                      (y1 + y2) / 2,
                      cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in cards]  # 计算中心坐标
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in cards]  # 计算两次的中心坐标距离
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])  # 找到距离最小的
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            result_image = cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id:  # 前后两次识别到的id相同，则进行搬运分类。若不同则重新识别
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 20:
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.is_moving = True
                    state.runner.start()
            else:
                state.moving_box = None
    return result_image


def image_proc_nums(img_in):
    outputs = yolov5_nums.detect(img_in)
    boxes, confs, classes = yolov5_nums.post_process(img_in, outputs, 0.95)
    cards = []
    width, height = IMAGE_SIZE
    result_image = img_in

    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        if cls_conf < 0.9:
            continue
        thr = TARGET_PIXEL_Y + (IMAGE_PIXEL_PRE_100MM_Y / 100.0 * 120.0)
        if y2 > thr:  # 如果坐标超出了识别范围(识别状态下吸盘正下发位置的像素坐标+80), 可能识别到错误的东西了，放弃掉
            continue
        cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        result_image = cv2.putText(img_in, NUMBERS_LABELS[cls_id] + " " + str(float(cls_conf))[:4],
                                   (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                   COLORS[CHARACTER_NUM + cls_id], 2)
        result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)),
                                     COLORS[CHARACTER_NUM + cls_id], 3)

    if state.is_moving:  # 机械臂在运动过程中不再进行新的运动
        return result_image

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = min(cards, key=lambda x: x[4])  # 识别到的所有卡牌中类别id最小的一个
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, cls_id, cls_conf  # 存起来
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            state.count = 0
        else:
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            cards = [((x1 + x2) / 2,
                      (y1 + y2) / 2,
                      cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in cards]  # 计算中心坐标
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in cards]  # 计算两次的中心坐标距离
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])  # 找到距离最小的
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            result_image = cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id:  # 前后两次识别到的id相同，则进行搬运分类。若不同则重新识别
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 20:
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.is_moving = True
                    state.runner.start()
            else:
                state.moving_box = None
    return result_image


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def image_proc():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    with state.lock:
        if state.running_mode == 1:
            result_img = image_proc_chars(image)
        elif state.running_mode == 2:
            result_img = image_proc_nums(image)
        else:
            result_img = image
    # fps cal
    fps_t1 = time.time()
    fps_cur = (1.0 / (fps_t1 - state.fps_t0))
    state.fps = fps_cur if state.fps == 0.0 else (state.fps * 0.8 + fps_cur * 0.2)
    state.fps_t0 = fps_t1
    show_fps(result_img, state.fps)
    #
    rgb_image = result_img.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass


def enter_func(msg):
    global TARGET_PIXEL_X, IMAGE_PIXEL_PRE_100MM_X
    global TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_Y

    rospy.loginfo("enter color palletizing")
    exit_func(msg)  # 先退出一次, 简化过程
    jetmax.go_home()
    state.reset()
    TARGET_PIXEL_X, TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/card',
        [TARGET_PIXEL_X,
         TARGET_PIXEL_Y,
         IMAGE_PIXEL_PRE_100MM_X,
         IMAGE_PIXEL_PRE_100MM_Y])
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)  # 订阅摄像头画面
    return TriggerResponse(success=True)


def exit_func(msg):
    rospy.loginfo("exit color palletizing")
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    with state.lock:
        state.running_mode = 0
        if isinstance(state.image_sub, rospy.Subscriber):  # 注销图像节点的订阅, 这样回调就不会被调用了
            rospy.loginfo('unregister image')
            state.image_sub.unregister()
            state.image_sub = None
    if isinstance(state.runner, threading.Thread):  # 若有码垛动作在运行等待码垛完成
        state.runner.join()
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    return TriggerResponse(success=True)


def heartbeat_timeout_cb():
    rospy.loginfo("heartbeat timeout. exiting...")
    rospy.ServiceProxy('/%s/exit' % ROS_NODE_NAME, Trigger)


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


def set_char_running_cb(msg):
    with state.lock:
        if msg.data:
            state.running_mode = 1
            state.moving_count = 0
        else:
            if state.running_mode == 1:
                state.running_mode = 0
    return [True, '']


def set_num_running_cb(msg):
    with state.lock:
        if msg.data:
            state.running_mode = 2
            state.moving_count = 0
        else:
            if state.running_mode == 2:
                state.running_mode = 0
    return [True, '']


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    TARGET_PIXEL_X, TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/card', [TARGET_PIXEL_X, TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y])
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # register result image pub
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    char_running_srv = rospy.Service('/%s/set_running_char' % ROS_NODE_NAME, SetBool, set_char_running_cb)
    num_running_srv = rospy.Service('/%s/set_running_num' % ROS_NODE_NAME, SetBool, set_num_running_cb)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)

    while True:
        try:
            image_proc()
            if rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            break
