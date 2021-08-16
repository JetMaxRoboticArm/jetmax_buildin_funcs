#!/usr/bin/env python3
import os
import sys
import rospy
import threading
import numpy as np
import pygame as pg
import threading
from functools import partial
import hiwonder

ROS_NODE_NAME = 'remote_control_joystick'
jetmax = None
sucker = None 
enable_control = True
os.environ["SDL_VIDEODRIVER"] = "dummy"  # For use PyGame without opening a visible display
pg.display.init()

PRESSED_ACTION_MAP = {}
HOLD_ACTION_MAP = {}
RELEASED_ACTION_MAP = {}
BUTTONS = ("CROSS", "CIRCLE", "None_1", "SQUARE",
           "TRIANGLE", "None_2", "L1", "R1",
           "L2", "R2", "SELECT", "START", "MODE",
           "L_HAT_LEFT", "L_HAT_RIGHT", "L_HAT_DOWN", "L_HAT_UP",
           "L_AXIS_LEFT", "L_AXIS_RIGHT", "L_AXIS_UP", "L_AXIS_DOWN",
           "R_AXIS_LEFT", "R_AXIS_RIGHT", "R_AXIS_UP", "R_AXIS_DOWN")

try:
    chassis = hiwonder.MecanumChassis()
except:
    chassis = None


servo1 = 90

def go_home():
    global enable_control, servo1
    enable_control = False
    jetmax.go_home(1)
    servo1 = 90
    hiwonder.pwm_servo1.set_position(servo1, 1)
    sucker.release()
    rospy.sleep(1)
    enable_control = True


def change_mode(new_mode):
    global PRESSED_ACTION_MAP, HOLD_ACTION_MAP, RELEASED_ACTION_MAP
    if new_mode == 0:
        hiwonder.buzzer.on()
        rospy.sleep(0.1)
        hiwonder.buzzer.off()
        PRESSED_ACTION_MAP = JOINT_MODE_PRESSED_ACTION_MAP
        HOLD_ACTION_MAP = JOINT_MODE_HOLD_ACTION_MAP
        RELEASED_ACTION_MAP = JOINT_RELEASED_ACTION_MAP

    elif new_mode == 1:
        hiwonder.buzzer.on()
        rospy.sleep(0.1)
        hiwonder.buzzer.off()
        rospy.sleep(0.05)
        hiwonder.buzzer.on()
        rospy.sleep(0.1)
        hiwonder.buzzer.off()
        PRESSED_ACTION_MAP = COORDINATE_MODE_PRESSED_ACTION_MAP
        HOLD_ACTION_MAP = COORDINATE_MODE_HOLD_ACTION_MAP
        RELEASED_ACTION_MAP = COORDINATE_MODE_RELEASED_ACTION_MAP
    else:
        pass

def set_pwm_servo1(v, dur):
    global servo1
    servo1 += v
    servo1 = servo1 if servo1 < 180 else 180
    servo1 = servo1 if servo1 > 0 else 0
    hiwonder.pwm_servo1.set_position(servo1, dur)

JOINT_MODE_PRESSED_ACTION_MAP = {
    "START": go_home,
    "SELECT": partial(change_mode, 1),
    "CIRCLE": partial(sucker.suck),
    "CROSS": partial(sucker.release),
    "L_HAT_LEFT": partial(jetmax.set_servo_relatively, 1, -2, 0.05),
    "L_HAT_RIGHT": partial(jetmax.set_servo_relatively, 1, 2, 0.05),
    "L_HAT_UP": partial(jetmax.set_servo_relatively, 2, 2, 0.05),
    "L_HAT_DOWN": partial(jetmax.set_servo_relatively, 2, -2, 0.05),
    "L1": partial(jetmax.set_servo_relatively, 3, -2, 0.05),
    "L2": partial(jetmax.set_servo_relatively, 3, 2, 0.05),
    "TRIANGLE": partial(set_pwm_servo1, -2, 0),
    "SQUARE": partial(set_pwm_servo1, 2, 0),
}

JOINT_MODE_HOLD_ACTION_MAP = {
    "L_HAT_LEFT": partial(jetmax.set_servo_relatively, 1, -8, 0.05),
    "L_HAT_RIGHT": partial(jetmax.set_servo_relatively, 1, 8, 0.05),
    "L_HAT_UP": partial(jetmax.set_servo_relatively, 2, 8, 0.05),
    "L_HAT_DOWN": partial(jetmax.set_servo_relatively, 2, -8, 0.05),
    "L1": partial(jetmax.set_servo_relatively, 3, -8, 0.05),
    "L2": partial(jetmax.set_servo_relatively, 3, 8, 0.05),
    "TRIANGLE": partial(set_pwm_servo1, -5, 0),
    "SQUARE": partial(set_pwm_servo1, 5, 0),
}

JOINT_RELEASED_ACTION_MAP = {
}

COORDINATE_MODE_PRESSED_ACTION_MAP = {
    "START": go_home,
    "SELECT": partial(change_mode, 0),
    "CIRCLE": partial(sucker.suck),
    "CROSS": partial(sucker.release),
    "TRIANGLE": partial(set_pwm_servo1, -2, 0),
    "SQUARE": partial(set_pwm_servo1, 2, 0),
    "L_HAT_LEFT": partial(jetmax.set_position_relatively, (-1, 0, 0), 0.05),
    "L_HAT_RIGHT": partial(jetmax.set_position_relatively, (1, 0, 0), 0.05),
    "L_HAT_UP": partial(jetmax.set_position_relatively, (0, 1, 0), 0.05),
    "L_HAT_DOWN": partial(jetmax.set_position_relatively, (0, -1, 0), 0.05),
    "L1": partial(jetmax.set_position_relatively, (0, 0, 1), 0.05),
    "L2": partial(jetmax.set_position_relatively, (0, 0, -1), 0.05),
}

COORDINATE_MODE_HOLD_ACTION_MAP = {
    "TRIANGLE": partial(set_pwm_servo1, -5, 0),
    "SQUARE": partial(set_pwm_servo1, 5, 0),
    "L_HAT_LEFT": partial(jetmax.set_position_relatively, (-4, 0, 0), 0.05),
    "L_HAT_RIGHT": partial(jetmax.set_position_relatively, (4, 0, 0), 0.05),
    "L_HAT_UP": partial(jetmax.set_position_relatively, (0, 4, 0), 0.05),
    "L_HAT_DOWN": partial(jetmax.set_position_relatively, (0, -4, 0), 0.05),
    "L1": partial(jetmax.set_position_relatively, (0, 0, 4), 0.05),
    "L2": partial(jetmax.set_position_relatively, (0, 0, -4), 0.05),
}

COORDINATE_MODE_RELEASED_ACTION_MAP = {
}


class Joystick:
    def __init__(self):
        self.js = None
        self.last_buttons = [0] * len(BUTTONS)
        self.hold_count = [0] * len(BUTTONS)
        self.lock = threading.Lock()
        self.speed_x = 0
        self.speed_y = 0
        threading.Thread(target=self.connect, daemon=True).start()

    def connect(self):
        while True:
            if os.path.exists("/dev/input/js0"):
                with self.lock:
                    if self.js is None:
                        pg.joystick.init()
                        try:
                            self.js = pg.joystick.Joystick(0)
                            self.js.init()
                        except Exception as e:
                            rospy.logerr(e)
                            self.js = None
            else:
                with self.lock:
                    if self.js is not None:
                        self.js.quit()
                        self.js = None
            rospy.sleep(0.2)

    def update_buttons(self):
        global enable_control
        with self.lock:
            if self.js is None or not enable_control:
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

        if chassis:
            axis[0] = hiwonder.misc.val_map(axis[0], -1, 1, -200, 200)
            axis[1] = hiwonder.misc.val_map(axis[1], -1, 1, 200, -200)
            axis[2] = hiwonder.misc.val_map(axis[2], -1, 1, 90, -90)
            v, d = chassis.translation(axis[0], axis[1], fake=True)
            chassis.set_velocity(v, d, axis[2])

        for i, button in enumerate(buttons_changed):
            if button:  # button state changed
                if buttons[i]:
                    rospy.logdebug(BUTTONS[i] + " pressed")
                    self.hold_count[i] = 0
                    button_name = BUTTONS[i]
                    if button_name in PRESSED_ACTION_MAP:
                        try:
                            PRESSED_ACTION_MAP[button_name]()
                        except Exception as e:
                                rospy.logerr(e)
                else:
                    rospy.logdebug(BUTTONS[i] + " released")
                    button_name = BUTTONS[i]
                    if button_name in RELEASED_ACTION_MAP:
                        try:
                            RELEASED_ACTION_MAP[button_name]()
                        except Exception as e:
                                rospy.logerr(e)
            else:
                if buttons[i]:
                    if self.hold_count[i] < 3:  # Better distinguish between short press and long press
                        self.hold_count[i] += 1
                    else:
                        rospy.logdebug(BUTTONS[i] + " hold")
                        button_name = BUTTONS[i]
                        if button_name in HOLD_ACTION_MAP:
                            try:
                                HOLD_ACTION_MAP[button_name]()
                            except Exception as e:
                                rospy.logerr(e)




def joystick_loop():
    js = Joystick()
    hiwonder.pwm_servo1.set_position(90, 1)
    sucker.release()
    rospy.sleep(1)
    change_mode(0)  # Joint mode as the default mode
    while True:
        try:
            js.update_buttons()
            rospy.sleep(0.05)
            if rospy.is_shutdown():
                sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO)
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    js = Joystick()
    jetmax.go_home(1)
    hiwonder.pwm_servo1.set_position(90, 1)
    sucker.release()
    rospy.sleep(1)
    change_mode(0)  # Joint mode as the default mode

    while True:
        try:
            js.update_buttons()
            rospy.sleep(0.05)
            if rospy.is_shutdown():
                sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
