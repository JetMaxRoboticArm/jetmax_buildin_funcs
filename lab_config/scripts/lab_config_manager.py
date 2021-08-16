#!/usr/bin/env python3
import os
import sys
import cv2
import rospy
import yaml
import numpy as np
from threading import RLock, Timer
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, SetBool, TriggerResponse, SetBoolResponse
from lab_config.srv import StashRange, GetRange, ChangeRange, GetAllColorName
from lab_config.srv import StashRangeResponse, GetRangeResponse, ChangeRangeResponse, GetAllColorNameResponse
from std_srvs.srv import Empty

lock = RLock()
image_sub = None
image_pub = None
kernel_erode = 3
kernel_dilate = 3
color_ranges = {}
current_range = {'min': [0, 0, 0], 'max': [100, 100, 100]}
config_file_path = os.path.join(sys.path[0], "../config/lab_config.yaml")
heartbeat_timer = None
sub_ed = False


def image_callback(ros_image):
    """
    callback of image_sub
    :params ros_image:
    """
    range_ = current_range
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    image_resize = cv2.resize(image, (400, 300), interpolation=cv2.INTER_NEAREST)
    frame_result = cv2.cvtColor(image_resize, cv2.COLOR_RGB2LAB)
    frame_result = cv2.GaussianBlur(frame_result, (3, 3), 3)
    mask = cv2.inRange(frame_result, tuple(range_['min']), tuple(range_['max']))  # 对原图像和掩模进行位运算
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_erode, kernel_erode)))
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_dilate, kernel_dilate)))

    rgb_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB).tostring()
    ros_image.data = rgb_image
    ros_image.height = 300
    ros_image.width = 400
    ros_image.step = ros_image.width * 3
    image_pub.publish(ros_image)


def enter_func(msg):
    rospy.loginfo(msg)
    global lock, image_sub, sub_ed
    with lock:
        if not sub_ed:
            image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
            sub_ed = True
            rospy.ServiceProxy('/usb_cam/start_capture', Empty)()
    return [True, '']


def exit_func(msg):
    rospy.loginfo(msg)
    global lock, image_sub, sub_ed
    with lock:
        try:
            image_sub.unregister()
            sub_ed = False
        except:
            pass

    return [True, '']



def set_running(msg):
    rospy.loginfo(msg)
    return [True, 'set_running']


def save_to_disk_srv_cb(msg):
    rospy.loginfo(msg)
    global lock, color_ranges

    with lock:
        cf = {"color_range_list": color_ranges.copy()}
        rospy.loginfo(cf)
    s = yaml.dump(cf, default_flow_style=False)
    with open(config_file_path, 'w') as f:
        f.write(s)
    rsp = TriggerResponse()
    rsp.success = True

    return rsp


def get_range_srv_cb(msg):
    rospy.loginfo(msg)
    global lock, color_ranges

    ranges = rospy.get_param('~color_range_list', color_ranges)
    rsp = GetRangeResponse()
    if msg.color_name in ranges:
        rsp.success = True
        rsp.min = ranges[msg.color_name]['min']
        rsp.max = ranges[msg.color_name]['max']
    else:
        rsp.success = False
    with lock:
        color_ranges = ranges

    return rsp


def change_range_srv_cb(msg):
    rospy.loginfo(msg)
    global current_range

    with lock:
        current_range = dict(min=list(msg.min), max=list(msg.max))
    rsp = ChangeRangeResponse()
    rsp.success = True

    return rsp


def stash_range_srv_cb(msg):
    rospy.loginfo(msg)
    global lock, color_ranges, current_range

    ranges = rospy.get_param('~color_range_list', color_ranges)
    with lock:
        ranges[msg.color_name] = current_range.copy()
    rospy.set_param('~color_range_list', ranges)
    color_ranges = ranges
    rsp = StashRangeResponse()
    rsp.success = True

    return rsp


def get_all_color_name_srv_cb(msg):
    rospy.loginfo(msg)
    global lock, color_ranges, current_range
    ranges = rospy.get_param('~color_range_list', color_ranges)
    color_names = list(ranges.keys())
    return GetAllColorNameResponse(color_names=color_names)

def heartbeat_timeout_cb():
    rospy.loginfo("heartbeat timeout. exiting...")
    rospy.ServiceProxy('/lab_config_manager/exit', Trigger)()

def heartbeat_srv_cb(msg):
    rospy.loginfo(msg)
    global heartbeat_timer
    if isinstance(heartbeat_timer, Timer):
        heartbeat_timer.cancel()
    rospy.logdebug("Heartbeat")
    if msg.data:
        heartbeat_timer = Timer(5, heartbeat_timeout_cb)
        heartbeat_timer.start()
    else:
        if isinstance(heartbeat_timer, Timer):
            heartbeat_timer.cancel()
    return SetBoolResponse(success=True)


if __name__ == '__main__':
    rospy.init_node('lab_config_manager', log_level=rospy.DEBUG)
    color_ranges = rospy.get_param('~color_range_list', {})
    kernel_erode = rospy.get_param('~kernel_erode', 3)
    kernel_dilate = rospy.get_param('~kernel_dilate', 3)
    config_file_path = rospy.get_param('~config_file_path', config_file_path)

    if 'red' in color_ranges:
        current_range = color_ranges['red']

    image_pub = rospy.Publisher('/lab_config_manager/image_result', Image, queue_size=1)
    enter_srv = rospy.Service('/lab_config_manager/enter', Trigger, enter_func)
    exit_srv = rospy.Service('/lab_config_manager/exit', Trigger, exit_func)
    running_srv = rospy.Service('/lab_config_manager/set_running', SetBool, set_running)

    save_to_disk_srv = rospy.Service('lab_config_manager/save_to_disk', Trigger, save_to_disk_srv_cb)
    get_color_range_srv = rospy.Service('lab_config_manager/get_range', GetRange, get_range_srv_cb)
    change_range_srv = rospy.Service('lab_config_manager/change_range', ChangeRange, change_range_srv_cb)
    stash_range_srv = rospy.Service('lab_config_manager/stash_range', StashRange, stash_range_srv_cb)
    get_all_color_name_srv = rospy.Service('/lab_config_manager/get_all_color_name', GetAllColorName,
                                           get_all_color_name_srv_cb)
    heartbeat_srv = rospy.Service('lab_config_manager/heartbeat', SetBool, heartbeat_srv_cb)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
