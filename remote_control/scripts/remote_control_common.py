import rospy
import hiwonder

ORIGIN_X, ORIGIN_Y, ORIGIN_Z = 0, 120, 100


class Sucker:
    def __init__(self, servo_id=1):
        self.state = False  # False for release, True for suck
        self.servo_id = servo_id
        self.servo = getattr(hiwonder, "pwm_servo{}".format(self.servo_id))

    def suck(self):
        # Can't set the speed to 100%, too high voltage may damage the motor
        hiwonder.motor1.set_speed(80)  # Turn on the air pump
        hiwonder.motor2.set_speed(0)  # Close the vent valve
        self.state = True

    def release(self, duration=0.5):
        # Can't set the speed to 100%, too high voltage may damage the valve
        hiwonder.motor1.set_speed(0)  # Turn off the air pump
        hiwonder.motor2.set_speed(80)  # Open the vent valve
        rospy.sleep(duration)
        hiwonder.motor1.set_speed(0)
        hiwonder.motor2.set_speed(0)
        self.state = False

    def set_angle(self, angle, duration):
        self.servo.set_position(int(angle + 90), int(duration))

    def change_angle(self, angle, duration):
        cur_angle = hiwonder.pwm_servo1.get_position() - 90
        angle += cur_angle
        angle = 90 if angle > 90 else angle
        angle = -90 if angle < -90 else angle
        self.set_angle(angle, duration)


class JetMax:
    def __init__(self, origin=None):
        self.origin = [ORIGIN_X, ORIGIN_Y, ORIGIN_Z] if origin is None else origin
        self.ik = hiwonder.kinematic.IKinematic()
        self.position = list(self.origin)
        self.servos = [500, 500, 500]  # [servo id 1, servo id 2, servo id 3]

    def move_to_pos(self, position, duration):
        duration = int(duration * 1000)
        x, y, z = position
        z = 220 if z > 220 else z
        z = 40 if z < 40 else z
        pos = self.ik.resolve(x, y, z)
        if pos is None:
            return False
        p1, p2, p3 = pos
        hiwonder.serial_servo.set_position(1, int(p1), duration)
        hiwonder.serial_servo.set_position(2, int(p2), duration)
        hiwonder.serial_servo.set_position(3, int(p3), duration)
        self.position = [x, y, z]
        self.servos = [int(p1), int(p2), int(p3)]
        return True

    def change_position(self, axis, value, duration):
        x, y, z = self.position
        if axis == 'x':
            x = x + value
        elif axis == 'y':
            y = y + value
        elif axis == 'z':
            z = z + value
        else:
            pass
        return self.move_to_pos((x, y, z), duration)

    def set_servo(self, servo_id, pulse, duration):
        if not 0 < servo_id < 4:
            return False
        pulse = int(pulse)
        duration = int(duration * 1000)
        hiwonder.serial_servo.set_position(servo_id, pulse, duration)
        self.position[servo_id - 1] = pulse
        return True

    def change_servo(self, servo_id, value, duration):
        if not 0 < servo_id < 4:
            return False
        index = servo_id - 1
        pos = self.servos[index]
        pos += value
        pos = 0 if pos < 0 else pos
        pos = 1000 if pos > 1000 else pos
        hiwonder.serial_servo.set_position(servo_id, pos, duration)
        self.servos[index] = pos
        return True

    def go_home(self, duration=1):
        return self.move_to_pos(self.origin, duration)

