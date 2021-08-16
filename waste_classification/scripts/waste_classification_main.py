#!/usr/bin/env python3
import os
import sys
import cv2
import math
import rospy
import numpy as np
import time
import threading
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
import queue
import hiwonder
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "waste_classification"

TRT_ENGINE_PATH = os.path.join(sys.path[0], "waste_v5_160.trt")
TRT_INPUT_SIZE = 160
TRT_CLASS_NAMES = ('Banana Peel', 'Broken Bones', 'Cigarette End', 'Disposable Chopsticks',
                   'Ketchup', 'Marker', 'Oral Liquid Bottle', 'Plate',
                   'Plastic Bottle', 'Storage Battery', 'Toothbrush', 'Umbrella')
TRT_NUM_CLASSES = 12
yolov5 = Yolov5TensorRT(TRT_ENGINE_PATH, TRT_INPUT_SIZE, TRT_NUM_CLASSES)
WASTE_CLASSES = {
    'food_waste': ('Banana Peel', 'Broken Bones', 'Ketchup'),
    'hazardous_waste': ('Marker', 'Oral Liquid Bottle', 'Storage Battery'),
    'recyclable_waste': ('Plastic Bottle', 'Toothbrush', 'Umbrella'),
    'residual_waste': ('Plate', 'Cigarette End', 'Disposable Chopsticks'),
}
COLORS = {
    'recyclable_waste': (0, 0, 255),
    'hazardous_waste': (255, 0, 0),
    'food_waste': (0, 255, 0),
    'residual_waste': (80, 80, 80)
}

GOAL_POSITION = {
    'recyclable_waste': (145, -65, 50, 65),
    'hazardous_waste': (140, -20, 50, 85),
    'food_waste': (140, 20, 50, 100),
    'residual_waste': (148, 65, 50, 115)
}

IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = 203.6, 230.5
TARGET_PIXEL_X, TARGET_PIXEL_Y = 380, 260
IMAGE_PROC_SIZE = 640, 480
image_queue = queue.Queue(maxsize=2)


class WasteClassification:
    def __init__(self):
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.is_moving = False
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.step = 0
        self.lock = threading.RLock()

    def reset(self):
        self.is_running = False
        self.moving_box = None
        self.is_moving = False
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.step = 0


state = WasteClassification()
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()





def moving():
    try:
        # 计算机械臂当前位置与目标位置的x, y 轴距离
        c_x, c_y, waste_class_name = state.moving_box
        cur_x, cur_y, cur_z = jetmax.position

        x = (TARGET_PIXEL_X - c_x) * (100.0 / IMAGE_PIXEL_PRE_100MM_X)
        y = (TARGET_PIXEL_Y - c_y) * (100.0 / IMAGE_PIXEL_PRE_100MM_Y)
        print("dist", x, y)

        # 计算当前位置与目标位置的欧氏距离，以控制运动速度
        t = math.sqrt(x * x + y * y + 140 * 140) / 140

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

        jetmax.set_position((new_x, new_y, 70), t)
        rospy.sleep(t+0.2)

        sucker.set_state(True)  # 启动气泵
        jetmax.set_position((new_x, new_y, 45), 0.5)  # 下移机械臂，吸取物品
        rospy.sleep(0.55)

        # 获取码垛的位置
        x, y, z, angle = GOAL_POSITION[waste_class_name]
        cur_x, cur_y, cur_z = jetmax.position  # 当前的机械臂位置
        jetmax.set_position((cur_x, cur_y, 140), 0.8)
        rospy.sleep(0.8)

        # 机械臂运动到码垛位置上空
        hiwonder.pwm_servo1.set_position(angle + arm_angle, 0.1)
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 160
        jetmax.set_position((x, y, 120), t)
        rospy.sleep(t)

        # 机械臂下移到目标码垛位置
        jetmax.set_position((x, y, z), 0.8)
        rospy.sleep(0.8)

        sucker.release(3)
        jetmax.set_position((x, y, z + 30), 0.6)  # 上提机械臂
        rospy.sleep(0.6)
        hiwonder.pwm_servo1.set_position(90, 0.5)  # 恢复吸盘角度

        # 回到初始位置
    finally:
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - jetmax.ORIGIN[0]) ** 2 + (cur_y - jetmax.ORIGIN[1]) ** 2) / 140
        hiwonder.pwm_servo1.set_position(90, 0.5)  # 恢复吸盘角度
        jetmax.go_home(t)
        rospy.sleep(t + 0.2)
        state.moving_box = None
        state.count = 0
        state.is_moving = False
        print("FINISHED")


def trt():
    global IMAGE_PIXEL_PRE_100MM_Y, TARGET_PIXEL_Y
    ros_image = image_queue.get(block=True)
    if state.is_running is False:
        image_pub.publish(ros_image)
        return
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    ta = time.time()
    outputs = yolov5.detect(image)
    boxes, confs, classes = yolov5.post_process(image, outputs, 0.85)
    width = image.shape[1]
    height = image.shape[0]
    cards = []
    for box, cls_conf, cls_id in zip(boxes, confs, classes):
        x1 = int(box[0] / TRT_INPUT_SIZE * width)
        y1 = int(box[1] / TRT_INPUT_SIZE * height)
        x2 = int(box[2] / TRT_INPUT_SIZE * width)
        y2 = int(box[3] / TRT_INPUT_SIZE * height)
        waste_name = TRT_CLASS_NAMES[cls_id]
        waste_class_name = ''
        for k, v in WASTE_CLASSES.items():
            if waste_name in v:
                waste_class_name = k
                break
        if cls_conf < 0.85:
            continue
        if y2 > TARGET_PIXEL_Y + (IMAGE_PIXEL_PRE_100MM_Y / 100.0 * 100):
            continue
        cards.append((cls_conf, x1, y1, x2, y2, waste_class_name))
        image = cv2.putText(image, waste_name + " " + str(float(cls_conf))[:4], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[waste_class_name], 3)

    if state.is_moving is False:
        if len(cards) == 0:
            state.count = 0
            state.moving_box = None
        else:
            if state.moving_box is None:
                moving_box = max(cards, key=lambda card: card[0])
                conf, x1, y1, x2, y2, waste_class_name = moving_box
                c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
                state.moving_box = c_x, c_y, waste_class_name
            else:
                l_c_x, l_c_y, l_waste_class_name = state.moving_box
                moving_box = min(cards, key=lambda card: math.sqrt((l_c_x - card[1]) ** 2 + (l_c_y - card[2]) ** 2))

                conf, x1, y1, x2, y2, waste_class_name = moving_box
                c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2

                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 6)
                image = cv2.circle(image, (int(c_x), int(c_y)), 1, (255, 255, 255), 10)

                if math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) > 30:
                    state.count = 0
                else:
                    c_x = l_c_x * 0.2 + c_x * 0.8
                    c_y = l_c_y * 0.2 + c_y * 0.8
                    state.count += 1
                state.moving_box = c_x, c_y, waste_class_name
                if state.count > 10:
                    print("AA")
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.is_moving = True
                    state.runner.start()
    rgb_image = image.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass


def enter_func(msg):
    global TARGET_PIXEL_X, TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y
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
    state.is_running = False
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    if isinstance(state.runner, threading.Thread):  # 若有码垛动作在运行等待码垛完成
        state.runner.join()
    if isinstance(state.image_sub, rospy.Subscriber):  # 注销图像节点的订阅, 这样回调就不会被调用了
        rospy.loginfo('unregister image')
        state.image_sub.unregister()
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    return TriggerResponse(success=True)


def set_running(msg: SetBoolRequest):
    if msg.data:
        rospy.loginfo("start running color palletizing")
        # 从参数服务器更新颜色阈值
        state.is_running = True
    else:
        rospy.loginfo("stop running object tracking")
        state.is_running = False
    return SetBoolResponse(success=True)


def heartbeat_timeout_cb():
    rospy.loginfo('heartbeat timeout. exiting...')
    rospy.ServiceProxy('/%s/exit' % ROS_NODE_NAME, Trigger)


def heartbeat_srv_cb(msg: SetBoolRequest):
    """
    心跳回调.会设置一个定时器，当定时到达后会调用退出服务退出玩法处理
    :params msg: 服务调用参数， std_srv SetBool
    """
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


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    TARGET_PIXEL_X, TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y = rospy.get_param(
        '/camera_cal/card', [TARGET_PIXEL_X, TARGET_PIXEL_Y, IMAGE_PIXEL_PRE_100MM_X, IMAGE_PIXEL_PRE_100MM_Y])
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # register result image pub
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)

    while True:
        try:
            trt()
            if rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            break
