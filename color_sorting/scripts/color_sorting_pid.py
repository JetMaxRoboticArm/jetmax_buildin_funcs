#!/usr/bin/env python3
import sys
import cv2
import math
import rospy
import numpy as np
import time
import threading
from sensor_msgs.msg import Image
from std_srvs.srv import *
import hiwonder.serial_servo as ssr
import hiwonder

range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

DEFAULT_X, DEFAULT_Y, DEFAULT_Z = 0, 138 + 8.14, 84 + 128.4
TRACKING_GOAL_X, TRACKING_GOAL_Y, TRACKING_GOAL_Z = 345, 320, 115
TRACKING_INTERVAL = 0.1
ROS_NODE_NAME = "color_sorting_1"
IMAGE_PROC_SIZE = 640, 480


class ColorSortingState:
    def __init__(self):
        # self.GREEN_GOLD =
        self.goal_position = {
            'green': ((157, 47, 80), -15),
            'red': ((0, 157, 100), 0),
            'blue': ((157, -47, 90), 15)
        }

        self.is_running = False
        self.is_moving = False, False
        self.moving_color = None
        self.target_colors = {
             'red': {
                 'min': [43, 158, 142],
                 'max': [255, 180, 202],
             },
            #'green': {
            #    'min': [23, 0, 136],
            #    'max': [125, 123, 161]
            #},
            # 'blue': {
            #     'min': [0, 0, 0],
            #     'max': [255, 255, 93],
            # }
        }
        self.lock = threading.RLock()
        self.position = 0, 0, 0
        self.runner = None
        self.count = 0
        self.step = 0
        self.x_pid = hiwonder.PID(0.07, 0.02, 0.001)
        self.y_pid = hiwonder.PID(0.04, 0.01, 0.0008)
        self.z_pid = hiwonder.PID(0.12, 0.10, 0.01)
        self.timestamp = time.time()

jetmax = hiwonder.JetMax()
stepper = hiwonder.Stepper(1)
stepper.go_home()
rospy.sleep(15)


state = ColorSortingState()
state.is_running = True
state.step = 1


cur_pos = 0
def go_to_pos(pos):
    global cur_pos
    stepper.set_div(stepper.DIV_1_8)
    offset = pos - cur_pos
    stepper.goto(offset)
    cur_pos = pos
    rospy.sleep(abs(offset/1000))


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
    hiwonder.pwm_servo1.set_position(90, 1)
    jetmax.go_home(1)
    rospy.sleep(1)
    hiwonder.motor2.set_speed(0)


def tracking_color():
    x, y, z = jetmax.position
    rect_x, rect_y = state.moving_color[0][0]

    dist_x = TRACKING_GOAL_X - rect_x
    state.x_pid.SetPoint = 0
    state.x_pid.update(dist_x)
    x -= state.x_pid.output

    rect_w = min(state.moving_color[0][1])
    dist_z = TRACKING_GOAL_Z - rect_w
    state.z_pid.update(dist_z)
    z += state.z_pid.output

    dist_y = TRACKING_GOAL_Y - rect_y
    state.y_pid.update(dist_y)
    y += state.y_pid.output

    jetmax.set_position((x, y, z), TRACKING_INTERVAL)
    state.timestamp = time.time() + TRACKING_INTERVAL
    if abs(dist_x) < 30 and abs(dist_y) < 30 and abs(dist_z) < 20:
        return True
    else:
        return False


def moving():
    rect, box, color_name = state.moving_color
    angle = rect[2]
    cur_x, cur_y, cur_z = jetmax.position
    dist = math.sqrt(cur_x * cur_x + cur_y * cur_y)
    print(dist)
    add_dist = 40  # Misc.map(dist, 80, 400, 30, 20)
    if cur_x == 0:
        y_ = add_dist
        x_ = 0
    else:
        k = cur_x / cur_y
        x_ = math.sin(math.atan(k)) * add_dist
        y_ = math.sqrt(x_ * x_ + add_dist * add_dist)

    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90

    hiwonder.pwm_servo1.set_position(90 + angle, 0.2)
    # move_to_pos(x_ + cur_x, y_ + cur_y, cur_z, 500)
    rospy.sleep(0.2)

    hiwonder.motor1.set_speed(100)
    cur_x, cur_y, cur_z = jetmax.position
    jetmax.set_position((cur_x, cur_y - 7, 18), 1) #get 70
    rospy.sleep(1.5)

    hiwonder.pwm_servo1.set_position(30, 0.8)
    jetmax.set_position((cur_x, cur_y - 7, 200), 1.2)
    rospy.sleep(1.2)

    (x, y, z), angle = state.goal_position['red']
    hiwonder.pwm_servo1.set_position(90 + angle, 0.8)
    if color_name == 'red':
        jetmax.set_position((-20, -150, 200), 1.5)
        go_to_pos(0)
    else:
        jetmax.set_position((40, -150, 200), 1.5)
        go_to_pos(10500)
    rospy.sleep(1.0)
    if color_name == 'red':
        jetmax.set_position((-20, -150, 70), 1.2)
    else:
        jetmax.set_position((40, -150, 70), 1.2)
    rospy.sleep(1.2)
    hiwonder.motor1.set_speed(0)
    hiwonder.motor2.set_speed(90)
    rospy.sleep(0.3)
    if color_name == 'red':
        jetmax.set_position((-20, -150, 200), 1)
    else:
        jetmax.set_position((20, -150, 200), 1)
    rospy.sleep(1)
    hiwonder.motor2.set_speed(0)

    jetmax.go_home(1.8)
    go_to_pos(5000)
    rospy.sleep(2)
    state.timestamp = time.time() + 2
    state.runner = None
    state.step = 1


def image_proc(img):
    img_h, img_w = img.shape[:2]

    frame_resize = cv2.resize(img, IMAGE_PROC_SIZE, interpolation=cv2.INTER_NEAREST)
    frame_lab = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2LAB)  # 将图像转换到LAB空间

    blocks = []
    if state.step == 0 or state.step == 1:
        for color_name, color in state.target_colors.items():
            img_gb = cv2.GaussianBlur(frame_lab, (5, 5), 5)
            frame_mask = cv2.inRange(img_gb, tuple(color['min']), tuple(color['max']))  # 对原图像和掩模进行位运算
            eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 腐蚀
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 膨胀
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出轮廓
            contour_area = map(lambda c: (c, math.fabs(cv2.contourArea(c))), contours)  # 计算各个轮廓的面积
            contour_area = list(filter(lambda c: c[1] > 200, contour_area))  # 剔除面积过小的轮廓

            if len(contour_area) > 0:
                for contour, area in contour_area:
                    rect = cv2.minAreaRect(contour)
                    center_x, center_y = rect[0]
                    center_x = int(hiwonder.misc.val_map(center_x, 0, IMAGE_PROC_SIZE[0], 0, img_w))  # 将缩放后的坐标缩放回原始图像大小
                    center_y = int(hiwonder.misc.val_map(center_y, 0, IMAGE_PROC_SIZE[1], 0, img_h))
                    box = np.int0(cv2.boxPoints(rect))  # 最小外接圆的四个顶点
                    for i, p in enumerate(box):
                        box[i, 0] = int(hiwonder.misc.val_map(p[0], 0, IMAGE_PROC_SIZE[0], 0, img_w))
                        box[i, 1] = int(hiwonder.misc.val_map(p[1], 0, IMAGE_PROC_SIZE[1], 0, img_h))
                    cv2.drawContours(img, [box], -1, range_rgb[color_name], 2)
                    cv2.circle(img, (center_x, center_y), 1, range_rgb[color_name], 5)
                    rect = list(rect)
                    rect[0] = (center_x, center_y)
                    blocks.append((rect, box, color_name))

        if len(blocks) > 0:
            if state.moving_color is None:
                state.moving_color = max(blocks, key=lambda tmp: tmp[0][1][0] * tmp[0][1][1])
            else:
                # print(state.moving_color)
                rect, _, _ = state.moving_color
                moving_x, moving_y = rect[0]
                blocks = list(map(lambda tmp: (tmp, math.sqrt((moving_x - tmp[0][0][0]) ** 2 +
                                                              (moving_y - tmp[0][0][1]) ** 2)), blocks))
                blocks.sort(key=lambda tmp: tmp[1])
                state.moving_color, _ = blocks[0]
                if state.step == 1:
                    cv2.drawContours(img, [state.moving_color[1]], -1, (255, 255, 255), 2)
                    cv2.circle(img, state.moving_color[0][0], 1, (255, 255, 255), 5)
                    if blocks[0][1] < 30:
                        state.count += 1
                        if state.count > 10 and time.time() > state.timestamp:
                            result = tracking_color()
                            # print(result)
                            if result is True:
                                state.step = 2
                                state.runner = threading.Thread(target=moving, daemon=True)
                                state.runner.start()
        else:
            if state.moving_color is not None:
                state.moving_color = None
                jetmax.go_home(1)
                state.timestamp = time.time() + 1.1
            state.count = 0

    cv2.line(img, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
    cv2.line(img, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)
    cv2.line(img, (390, 0), (390, 479), (0, 255, 255), 2)
    return img


def image_callback(ros_image):
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame_result = img_bgr.copy()
    with state.lock:
        if state.is_running:
            frame_result = image_proc(frame_result)
    cv2.imshow("IMG", frame_result)
    cv2.waitKey(1)
    #rgb_image = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB).tostring()
    #ros_image.data = rgb_image
    #image_pub.publish(ros_image)


def enter_func(msg):
    global lock, image_sub, __isRunning, org_image_sub_ed
    rospy.loginfo("enter object tracking")
    with lock:
        init()
        if not org_image_sub_ed:
            org_image_sub_ed = True
            image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    return TriggerResponse()


def exit_func(msg):
    global lock, image_sub, __isRunning, org_image_sub_ed
    rospy.loginfo("exit object tracking")
    with lock:
        __isRunning = False
        try:
            if org_image_sub_ed:
                org_image_sub_ed = False
                image_sub.unregister()
        except:
            pass
    return TriggerResponse()


def start_running():
    global lock, __isRunning
    rospy.loginfo("start running object tracking")
    with lock:
        __isRunning = True


def stop_running():
    global lock, __isRunning
    rospy.loginfo("stop running object tracking")
    with lock:
        __isRunning = False
        reset()


def set_running(msg):
    if msg.data:
        start_running()
    else:
        stop_running()
    return SetBoolResponse()


def set_target(msg):
    global lock, __target_color
    with lock:
        if '#' in msg.data:
            __target_color = msg.data[1:]
        else:
            __target_color = msg.data
    return SetTargetResponse()


if __name__ == '__main__':
    go_to_pos(5200)
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    rospy.sleep(0.2)
    init()
    color_ranges = rospy.get_param('/lab_config_manager/color_range_list', {})
    state.target_colors = {
            'red': color_ranges['red'],
            'blue': color_ranges['blue'],
            }
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image,
                                queue_size=1)  # register result image publisher
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        sys.exit(0)
