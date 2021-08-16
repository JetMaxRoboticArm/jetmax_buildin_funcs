#!/usr/bin/env python3
import actionlib
import rospy
from jetmax_control.msg import Mecanum
from jetmax_control.msg import JetMax as JetMaxState
from jetmax_control.msg import SetServo, SetJoint, SetJetMax
from std_msgs.msg import UInt8, UInt16, Float32, Bool
from std_srvs.srv import Empty
import hiwonder
import threading
import action_set

# import jetmax_control_joystick as joystick


class JetMaxControl:
    def __init__(self):
        rospy.init_node('jetmax_control', anonymous=True)
        self.pwm_servos = [hiwonder.pwm_servo1, hiwonder.pwm_servo2]

        """
        Mecanum topics
        """
        try:
            self.chassis = hiwonder.MecanumChassis()
            self.mecanum_pub = rospy.Publisher('/jetmax/mecanum/status', Mecanum, queue_size=1)
            self.mecanum_sub = rospy.Subscriber('/jetmax/mecanum/command', Mecanum,
                                                queue_size=1,
                                                callback=lambda msg: self.chassis.set_velocity(msg.velocity,
                                                                                               msg.direction,
                                                                                               msg.angular_rate))
        except Exception:
            self.chassis = None
            self.mecanum_sub = None
            self.mecanum_pub = None

        """
        Arm topics
        """
        self.jetmax = hiwonder.JetMax()
        self.jetmax_pub = rospy.Publisher('/jetmax/status', JetMaxState, queue_size=1)
        self.jetmax_sub = rospy.Subscriber('/jetmax/command', SetJetMax,
                                           queue_size=1,
                                           callback=lambda msg: self.jetmax.set_position((msg.x, msg.y, msg.z),
                                                                                         msg.duration))
        self.jetmax_relative_sub = rospy.Subscriber('/jetmax/relative_command', SetJetMax,
                                                    queue_size=1,
                                                    callback=lambda msg: self.jetmax.set_position_relatively(
                                                        (msg.x, msg.y, msg.z), msg.duration))

        self.jetmax_joint_pubs = []
        self.jetmax_joint_subs = []
        self.jetmax_servo_pubs = []
        self.jetmax_servo_subs = []
        for i in range(1, 4):
            joint_pub = rospy.Publisher('/jetmax/joint{}/status'.format(i), Float32, queue_size=1)
            joint_cmd = rospy.Subscriber('/jetmax/joint{}/command'.format(i), SetJoint,
                                         queue_size=1,
                                         callback_args=i,
                                         callback=lambda msg, joint_id: self.jetmax.set_joint(joint_id,
                                                                                              msg.data,
                                                                                              msg.duration))
            servo_pub = rospy.Publisher('/jetmax/servo{}/status'.format(i), UInt16, queue_size=1)
            servo_cmd = rospy.Subscriber('/jetmax/servo{}/command'.format(i), SetServo,
                                         queue_size=1,
                                         callback_args=i,
                                         callback=lambda msg, servo_id: self.jetmax.set_servo(servo_id,
                                                                                              msg.data,
                                                                                              msg.duration))
            self.jetmax_joint_pubs.append(joint_pub)
            self.jetmax_joint_subs.append(joint_cmd)
            self.jetmax_servo_pubs.append(servo_pub)
            self.jetmax_servo_subs.append(servo_cmd)

        """
        End effector topics
        """
        self.end_effector_servo_pubs = []
        self.end_effector_servo_subs = []
        for i in range(1, 3):
            pub = rospy.Publisher('/jetmax/end_effector/servo{}/status'.format(i), UInt8, queue_size=1)
            cmd = rospy.Subscriber('/jetmax/end_effector/servo{}/command'.format(i), SetServo,
                                   queue_size=1,
                                   callback_args=i - 1,
                                   callback=lambda msg, servo_id: self.pwm_servos[servo_id].set_position(msg.data,
                                                                                                         msg.duration))
            self.end_effector_servo_pubs.append(pub)
            self.end_effector_servo_subs.append(cmd)

        self.sucker = hiwonder.Sucker()
        self.sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/status', Bool, queue_size=1)
        self.sucker_sub = rospy.Subscriber('/jetmax/end_effector/sucker/command', Bool,
                                           queue_size=1,
                                           callback=lambda msg: self.sucker.set_state(msg.data))

        self.go_home_srv = rospy.Service('/jetmax/go_home', Empty, self.go_home)
        self.action_set = action_set.ActionSetRunner(self.jetmax, self.sucker)

        self.rate = rospy.Rate(10)  # 10hz
        self.go_home()
        hiwonder.buzzer.on()
        rospy.sleep(0.1)
        hiwonder.buzzer.off()


    def go_home(self, req=None):
        self.jetmax.go_home(2)
        return []

    def update(self):
        while not rospy.is_shutdown():
            """
            Update chassis status
            """
            if self.chassis:
                mecanum_status = Mecanum(
                    velocity=self.chassis.velocity,
                    direction=self.chassis.direction,
                    angular_rate=self.chassis.angular_rate
                )
                self.mecanum_pub.publish(mecanum_status)

            """
            Update end effector status
            """
            pwm_servo_states = []
            for i in range(2):
                servo_state = UInt8(data=int(self.pwm_servos[i].get_position()))
                self.end_effector_servo_pubs[i].publish(servo_state)
                pwm_servo_states.append(servo_state)
            self.sucker_pub.publish(self.sucker.get_state())

            """
            Update arm status
            """
            servo_states = []
            joint_states = []
            for i in range(3):
                servo_state = int(self.jetmax.servos[i])
                joint_state = self.jetmax.joints[i]
                self.jetmax_servo_pubs[i].publish(UInt16(data=servo_state))
                self.jetmax_joint_pubs[i].publish(Float32(data=joint_state))
                servo_states.append(servo_state)
                joint_states.append(joint_state)
            x, y, z = self.jetmax.position
            self.jetmax_pub.publish(JetMaxState(
                x=x, y=y, z=z,
                joint1=joint_states[0], joint2=joint_states[1], joint3=joint_states[2],
                servo1=servo_states[0], servo2=servo_states[1], servo3=servo_states[2],
                pwm1=hiwonder.pwm_servo1.get_position(), pwm2=hiwonder.pwm_servo2.get_position(), sucker=self.sucker.get_state()
            ))

            self.rate.sleep()


if __name__ == '__main__':
    jetmax = JetMaxControl()
    # joystick.jetmax = jetmax.jetmax
    # joystick.sucker = jetmax.sucker
    try:
        #    threading.Thread(target=joystick.joystick_loop, daemon=True).start()
        jetmax.update()
    except rospy.ROSInterruptException:
        pass
