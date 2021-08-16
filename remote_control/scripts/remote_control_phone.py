#!/usr/bin/env python3
import os
import sys
import rospy
import math
import threading
from functools import partial
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from remote_control.srv import ChangePosition, ChangePositionResponse, ChangePositionRequest
from remote_control.srv import SetPosition, SetPositionResponse, SetPWMServoRequest
from remote_control.srv import SetChuck, SetChuckResponse, SetChuckRequest
from remote_control.srv import SetPosition, SetChuckResponse, SetPositionRequest
from remote_control.srv import SetPWMServo, SetPWMServoResponse, SetPWMServoRequest
from remote_control.srv import SetServo, SetServoResponse, SetServoRequest
from remote_control.msg import Status
import hiwonder
from hiwonder import serial_servo as ss
import pygame as pg
import numpy as np

ROS_NODE_NAME = 'remote_control'
DEFAULT_X, DEFAULT_Y, DEFAULT_Z = 0, 138 + 8.14, 84 + 128.4


class RemoteControl:
    def __init__(self):
        self.position = DEFAULT_X, DEFAULT_Y, DEFAULT_X
        self.current_angle = [90, 90]
        self.heartbeat_timer = None
        self.servos = [500, 500, 500, 500]

    def reset(self):
        self.position = DEFAULT_X, DEFAULT_Y, DEFAULT_X
        self.current_angle = [90, 90]


ikinematic = hiwonder.kinematic.IKinematic()
state = RemoteControl()


def move_to_pos(x, y, z, t):
    if z > 240:
        z = 240
    if z < 40:
        z = 40
    pos = ikinematic.resolve(x, y, z)
    if pos is None:
        return
    p1, p2, p3 = pos
    ss.set_position(1, int(p1), t)
    ss.set_position(2, int(p2), t)
    ss.set_position(3, int(p3), t)
    state.position = x, y, z
    state.servos = [int(p1), int(p2), int(p3), state.servos[3]]


def init():
    state.reset()
    hiwonder.motor1.set_speed(0)
    hiwonder.motor2.set_speed(100)
    hiwonder.pwm_servo1.set_position(90, 1000)
    hiwonder.pwm_servo2.set_position(90, 1000)
    move_to_pos(DEFAULT_X, DEFAULT_Y, DEFAULT_Z, 1000)
    rospy.sleep(1)
    hiwonder.motor2.set_speed(0)


def enter_func(msg):
    rospy.loginfo('enter remote control')
    exit_func(msg)  # 先退出一次, 简化过程
    init()  # 初始化位置及状态
    rospy.ServiceProxy('/usb_cam/start_capture', Empty)()
    return TriggerResponse(success=True, message='')


def exit_func(msg):
    rospy.loginfo('exit remote control')
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    # rospy.ServiceProxy('/usb_cam/stop_capture', Empty)()
    return TriggerResponse(success=True, message='')


def set_running(msg: SetBoolRequest):
    rospy.loginfo('set running')
    return SetBoolResponse(success=True, message='')


def heartbeat_timeout_cb():
    rospy.loginfo('heartbeat timeout. exiting...')
    rospy.ServiceProxy('/%s/exit' % ROS_NODE_NAME, Trigger)()


def heartbeat_srv_cb(msg: SetBoolRequest):
    """
    心跳回调.会设置一个定时器，当定时到达后会调用退出服务退出玩法处理
    :params msg: 服务调用参数， std_srv SetBool
    """
    if isinstance(state.heartbeat_timer, threading.Timer):
        state.heartbeat_timer.cancel()
    rospy.logdebug("Heartbeat " + str(msg))
    if msg.data:
        state.heartbeat_timer = threading.Timer(5, heartbeat_timeout_cb)
        state.heartbeat_timer.start()
    else:
        state.heartbeat_timer.cancel()
    return SetBoolResponse(success=msg.data, message='')


def go_home_cb(msg: TriggerResponse):
    rospy.logdebug(msg)
    init()  # 初始化位置及状态
    return TriggerResponse(success=True, message='')


def change_position_cb(msg: ChangePositionRequest):
    rospy.logdebug(msg)
    x, y, z = state.position
    if msg.axis_name == 'x':
        x = x + msg.change_value
    elif msg.axis_name == 'y':
        y = y + msg.change_value
    elif msg.axis_name == 'z':
        z = z + msg.change_value
    else:
        pass
    rospy.logdebug((x, y, z))
    move_to_pos(x, y, z, int(msg.duration * 1000))
    return ChangePositionResponse(success=True)


def set_position_cb(msg: SetPositionRequest):
    rospy.logdebug(msg)
    move_to_pos(msg.x, msg.y, msg.z, int(msg.duration * 1000))
    return SetPositionResponse(success=True)


def set_chuck_cb(msg: SetChuckRequest):
    rospy.logdebug(msg)
    if msg.absorb is True:
        hiwonder.motor1.set_speed(100)
        hiwonder.motor2.set_speed(0)
    else:
        hiwonder.motor1.set_speed(0)
        hiwonder.motor2.set_speed(100)
        rospy.sleep(0.1)
        hiwonder.motor2.set_speed(0)
    return SetChuckResponse(success=True, message='')


def set_pwm_servo_cb(msg: SetPWMServoRequest):
    state.current_angle[msg.servo_id - 1] = msg.angle
    hiwonder.pwm_servo1.set_position(state.current_angle[0], int(msg.duration * 1000))
    hiwonder.pwm_servo2.set_position(state.current_angle[1], int(msg.duration * 1000))
    return SetPWMServoResponse(success=True, message='')


def set_serial_servo_cb(msg: SetServoRequest):
    ss.set_position(int(msg.servo_id), int(msg.angle), int(msg.duration * 1000))
    return SetServoResponse(success=True, message='')


def change_pwm_servo_cb(msg: SetPWMServoRequest):
    state.current_angle[msg.servo_id - 1] += msg.angle
    if state.current_angle[msg.servo_id - 1] > 180:
        state.current_angle[msg.servo_id - 1] = 180
    if state.current_angle[msg.servo_id - 1] < 0:
        state.current_angle[msg.servo_id - 1] = 0
    hiwonder.pwm_servo1.set_position(state.current_angle[0], int(msg.duration * 1000))
    hiwonder.pwm_servo2.set_position(state.current_angle[1], int(msg.duration * 1000))
    return SetPWMServoResponse(success=True, message='')


def rolling(angle, duration):
    x, y, z = state.position
    angle = math.radians(angle)
    x1 = math.cos(angle) * x - math.sin(angle) * y
    y1 = math.sin(angle) * x + math.cos(angle) * y
    move_to_pos(x1, y1, z, duration)


BUTTON_MAP = ("CROSS", "CIRCLE", "None_1", "SQUARE",
              "TRIANGLE", "None_2", "L1", "R1",
              "L2", "R2", "SELECT", "START", "MODE",
              "L_HAT_LEFT", "L_HAT_RIGHT", "L_HAT_DOWN", "L_HAT_UP",
              "L_AXIS_LEFT", "L_AXIS_RIGHT", "L_AXIS_UP", "L_AXIS_DOWN",
              "R_AXIS_LEFT", "R_AXIS_RIGHT", "R_AXIS_UP", "R_AXIS_DOWN")

PRESSED_ACTION_MAP = {
    'START': init,
    'R1': partial(set_chuck_cb, SetChuckRequest(absorb=True)),
    'R2': partial(set_chuck_cb, SetChuckRequest(absorb=False)),
    'TRIANGLE': partial(set_chuck_cb, SetChuckRequest(absorb=True)),
}

HOLD_ACTION_MAP = {
    'L_HAT_LEFT': partial(change_position_cb, ChangePositionRequest(axis_name='x', change_value=-7, duration=0.05)),
    'L_HAT_RIGHT': partial(change_position_cb, ChangePositionRequest(axis_name='x', change_value=7, duration=0.05)),
    'L_HAT_UP': partial(change_position_cb, ChangePositionRequest(axis_name='y', change_value=7, duration=0.05)),
    'L_HAT_DOWN': partial(change_position_cb, ChangePositionRequest(axis_name='y', change_value=-7, duration=0.05)),
    'L1': partial(change_position_cb, ChangePositionRequest(axis_name='z', change_value=4, duration=0.05)),
    'L2': partial(change_position_cb, ChangePositionRequest(axis_name='z', change_value=-4, duration=0.05)),
    # 'SQUARE': partial(change_pwm_servo_cb, SetPWMServoRequest(angle=-5, duration=0)),
    # 'CIRCLE': partial(change_pwm_servo_cb, SetPWMServoRequest(angle=5, duration=0)),
    'L_AXIS_LEFT': partial(rolling, angle=1, duration=0.05),
    'L_AXIS_RIGHT': partial(rolling, angle=-1, duration=0.05),
    'R_AXIS_LEFT': partial(change_pwm_servo_cb, SetPWMServoRequest(angle=-5, duration=0)),
    'R_AXIS_RIGHT': partial(change_pwm_servo_cb, SetPWMServoRequest(angle=5, duration=0)),
    'CIRCLE': partial(set_pwm_servo_cb, SetPWMServoRequest(angle=70, duration=0)),
    'SQUARE': partial(set_pwm_servo_cb, SetPWMServoRequest(angle=60, duration=0)),
    'CROSS': partial(set_pwm_servo_cb, SetPWMServoRequest(angle=30, duration=0)),
}

RELEASED_ACTION_MAP = {
}


class Joystick:
    def __init__(self):
        self.js = None
        self.last_buttons = [0] * len(BUTTON_MAP)
        self._lock = threading.Lock()
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pg.display.init()
        threading.Thread(target=self.connect, daemon=True).start()

    def connect(self):
        while True:
            if os.path.exists("/dev/input/js0"):
                with self._lock:
                    if self.js is None:
                        pg.joystick.init()
                        try:
                            self.js = pg.joystick.Joystick(0)
                            self.js.init()
                        except Exception as e:
                            rospy.logerr(e)
                            self.js = None
            else:
                with self._lock:
                    if self.js is not None:
                        self.js.quit()
                        self.js = None
            rospy.sleep(0.2)

    def update_buttons(self):
        with self._lock:
            if self.js is None:
                return

            # update and read joystick data
            pg.event.pump()
            buttons = [self.js.get_button(i) for i in range(13)]
            hat = list(self.js.get_hat(0))
            axis = [self.js.get_axis(i) for i in range(4)]
            hat.extend(axis)

            # convert analog data to digital
            for i in range(6):
                buttons.extend([1 if hat[i] < -0.5 else 0, 1 if hat[i] > 0.5 else 0])

            # check what has changed in this update
            buttons = np.array(buttons)
            buttons_changed = np.bitwise_xor(self.last_buttons, buttons).tolist()
            self.last_buttons = buttons  # save buttons data

        for i, button in enumerate(buttons_changed):
            if button:  # button state changed
                if buttons[i]:
                    rospy.logdebug(BUTTON_MAP[i] + " pressed")
                    button_name = BUTTON_MAP[i]
                    if button_name in PRESSED_ACTION_MAP:
                        PRESSED_ACTION_MAP[button_name]()
                else:
                    rospy.logdebug(BUTTON_MAP[i] + " released")
                    button_name = BUTTON_MAP[i]
                    if button_name in RELEASED_ACTION_MAP:
                        RELEASED_ACTION_MAP[button_name]()
            else:
                if buttons[i]:
                    button_name = BUTTON_MAP[i]
                    if button_name in HOLD_ACTION_MAP:
                        HOLD_ACTION_MAP[button_name]()


def status_pub_cb(v):
    x, y, z = state.position
    x, y, z = int(x), int(y), int(z)
    id1, id2, id3, id4 = state.servos
    pwm1 = 90
    motor1, motor2 = 100, 100
    status_pub.publish(x=x,
                       y=y,
                       z=z,
                       id1=id1,
                       id2=id2,
                       id3=id3,
                       id4=id4,
                       pwm1=pwm1,
                       motor1=motor1,
                       motor2=motor2)


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    init()
    js = Joystick()

    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)

    go_home_srv = rospy.Service('/%s/go_home' % ROS_NODE_NAME, Trigger, go_home_cb)
    change_position_srv = rospy.Service('/%s/change_position' % ROS_NODE_NAME, ChangePosition, change_position_cb)
    set_position_srv = rospy.Service('/%s/set_position' % ROS_NODE_NAME, SetPosition, set_position_cb)
    set_chuck_srv = rospy.Service('/%s/set_chuck' % ROS_NODE_NAME, SetChuck, set_chuck_cb)
    set_pwm_servo_srv = rospy.Service('/%s/set_pwm_servo' % ROS_NODE_NAME, SetPWMServo, set_pwm_servo_cb)
    set_serial_servo_srv = rospy.Service('/%s/set_serial_servo' % ROS_NODE_NAME, SetServo, set_serial_servo_cb)

    running_srv = rospy.Service('/%s/set_running' % ROS_NODE_NAME, SetBool, set_running)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)

    status_pub = rospy.Publisher('/%s/status' % ROS_NODE_NAME, Status, queue_size=1)
    status_pub_timer = rospy.Timer(rospy.Duration(1.0 / 5.0), status_pub_cb)

    while True:
        try:
            rospy.sleep(0.05)
            js.update_buttons()
            if rospy.is_shutdown():
                sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
