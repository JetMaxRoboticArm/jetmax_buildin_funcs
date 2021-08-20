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
import hiwonder
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from jetmax_control.msg import SetServo
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "alphabetically"
IMAGE_SIZE = 640, 480

CHARACTERS_ENGINE_PATH = os.path.join(sys.path[0], 'characters_v5_160.trt')
CHARACTER_LABELS = tuple([i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
CHARACTER_NUM = 26

NUMBERS_ENGINE_PATH = os.path.join(sys.path[0], 'numbers_v5_160.trt')
NUMBERS_LABELS = tuple([i for i in '0123456789+-*/='])
NUMBERS_NUM = 15

TRT_INPUT_SIZE = 160
COLORS = tuple([tuple([random.randint(10, 255) for j in range(3)]) for i in range(CHARACTER_NUM + NUMBERS_NUM)])

GOAL_POSITIONS = (
    (-225, -90, 55, 17), (-225, -45, 55, 7), (-225, 0, 55, -3), (-225, 40, 55, -13),
    (-180, -90, 55, 20), (-180, -45, 55, 10), (-180, 0, 55, -3), (-180, 40, 55, -13),
    (-135, -90, 55, 30), (-135, -45, 55, 15), (-135, 0, 55, -5), (-135, 40, 55, -18),
)


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


class Alphabetically:
    def __init__(self):
        self.running_mode = 0
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.moving_count = 0
        self.count = 0
        self.lock = threading.RLock()
        self.fps_t0 = time.time()
        self.fps = 0
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None

    def reset(self):
        self.running_mode = 0
        self.moving_box = None
        self.moving_count = 0
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.fps_t0 = time.time()
        self.fps = 0

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/card_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)


def moving():
    try:
        c_x, c_y, cls_id, cls_conf = state.moving_box
        cur_x, cur_y, cur_z = jetmax.position

        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
        t = math.sqrt(x * x + y * y + 120 * 120) / 120
        new_x, new_y = cur_x + x, cur_y + y
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass
        jetmax.set_position((new_x, new_y, 70), t)
        rospy.sleep(t + 0.5)

        sucker.set_state(True)
        jetmax.set_position((new_x, new_y, 50), 0.8)
        rospy.sleep(0.85)

        x, y, z, angle = GOAL_POSITIONS[state.moving_count]
        cur_x, cur_y, cur_z = jetmax.position
        jetmax.set_position((cur_x, cur_y, 100), 0.8)
        rospy.sleep(0.8)

        hiwonder.pwm_servo1.set_position(90 + angle + arm_angle, 0.1)
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 150
        jetmax.set_position((x, y, z + 30), t)
        rospy.sleep(t)

        jetmax.set_position((x, y, z), 0.8)
        rospy.sleep(0.8)

        sucker.release(3)
        jetmax.set_position((x, y, z + 30), 0.8)
        rospy.sleep(0.1)
        hiwonder.pwm_servo1.set_position(90, 0.4)
        rospy.sleep(0.8)

    finally:
        sucker.release(3)
        cur_x, cur_y, cur_z = jetmax.position
        t = math.sqrt((cur_x - jetmax.ORIGIN[0]) ** 2 + (cur_y - jetmax.ORIGIN[1]) ** 2) / 120
        jetmax.go_home(t)
        hiwonder.pwm_servo1.set_position(90, 0.2)
        rospy.sleep(t + 0.2)
        with state.lock:
            state.moving_box = None
            state.moving_count += 1
            if state.moving_count >= len(GOAL_POSITIONS):
                state.moving_count = 0
            state.runner = None
        print("FINISHED")


def image_proc_chars(img_in):
    if state.runner is not None:
        return img_in
    result_image = img_in
    outputs = yolov5_chars.detect(np.copy(img_in))
    boxes, confs, classes = yolov5_chars.post_process(img_in, outputs, 0.90)
    cards = []
    width, height = IMAGE_SIZE
    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        if cls_conf < 0.9:
            continue
        cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        cv2.putText(img_in, CHARACTER_LABELS[cls_id] + " " + str(float(cls_conf))[:4],
                    (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls_id], 2)
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls_id], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = min(cards, key=lambda x: x[4])
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, cls_id, cls_conf
            cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            state.count = 0
        else:
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            cards = [((x1 + x2) / 2,
                      (y1 + y2) / 2,
                      cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in cards]
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in cards]
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id:
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 20:
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.runner.start()
            else:
                state.moving_box = None
    return result_image


def image_proc_nums(img_in):
    if state.runner is not None:
        return img_in
    result_image = img_in
    outputs = yolov5_nums.detect(np.copy(img_in))
    boxes, confs, classes = yolov5_nums.post_process(img_in, outputs, 0.95)
    cards = []
    width, height = IMAGE_SIZE

    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        if cls_conf < 0.9:
            continue
        cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        result_image = cv2.putText(img_in, NUMBERS_LABELS[cls_id] + " " + str(float(cls_conf))[:4],
                                   (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                   COLORS[CHARACTER_NUM + cls_id], 2)
        result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)),
                                     COLORS[CHARACTER_NUM + cls_id], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = min(cards, key=lambda x: x[4])
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, cls_id, cls_conf
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            state.count = 0
        else:
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            cards = [((x1 + x2) / 2,
                      (y1 + y2) / 2,
                      cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in cards]
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in cards]
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            result_image = cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id:
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 20:
                    state.runner = threading.Thread(target=moving, daemon=True)
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
    # show_fps(result_img, state.fps)
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
    rospy.loginfo("enter")
    exit_func(msg)
    jetmax.go_home()
    state.reset()
    state.load_camera_params()
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    return TriggerResponse(success=True)


def exit_func(msg):
    rospy.loginfo("exit")
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    with state.lock:
        state.running_mode = 0
        if isinstance(state.image_sub, rospy.Subscriber):
            rospy.loginfo('unregister image')
            state.image_sub.unregister()
            state.image_sub = None
    if isinstance(state.runner, threading.Thread):
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
    state = Alphabetically()
    state.load_camera_params()
    if state.camera_params is None:
        rospy.logerr("Can not load camera parameters")
        sys.exit(-1)
    yolov5_chars = Yolov5TensorRT(CHARACTERS_ENGINE_PATH, TRT_INPUT_SIZE, CHARACTER_NUM)
    yolov5_nums = Yolov5TensorRT(NUMBERS_ENGINE_PATH, TRT_INPUT_SIZE, NUMBERS_NUM)
    image_queue = queue.Queue(maxsize=1)
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()

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
