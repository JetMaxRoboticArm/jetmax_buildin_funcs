import asyncio
import threading
import hiwonder
import os
import json
import actionlib
from jetmax_control.msg import ActionSetRawAction, ActionSetRawFeedback, ActionSetRawActionGoal, ActionSetRawResult
from jetmax_control.srv import ActionSetList, ActionSetListResponse
from jetmax_control.srv import ActionSetFileOp, ActionSetFileOpResponse
from std_srvs.srv import Trigger
import rospy


class ActionSetRunner:
    def __init__(self, jetmax, sucker):
        self._running = None
        self._loop = asyncio.new_event_loop()
        self._lock = threading.RLock()
        self._jetmax = jetmax
        self._sucker = sucker
        self.action_set_online = actionlib.SimpleActionServer('/jetmax/actionset_online', ActionSetRawAction,
                                                              self.run_online,
                                                              auto_start=False)
        threading.Thread(target=self._start_loop, daemon=True).start()
        self.action_set_online.start()
        self.get_actionset_list_srv = rospy.Service('/jetmax/actionset/get_actionset_list', ActionSetList,
                                                    self.get_action_list_cb)
        self.remove_actionset_srv = rospy.Service('/jetmax/actionset/remove_actionset', ActionSetFileOp,
                                                  self.remove_actionset_cb)
        self.save_actionset_srv = rospy.Service('/jetmax/actionset/save_actionset', ActionSetFileOp,
                                                self.save_actionset_cb)
        self.read_actionset_srv = rospy.Service('/jetmax/actionset/read_actionset', ActionSetFileOp,
                                                self.read_actionset_cb)

    def get_action_list_cb(self, req):
        for root, dirs, files in os.walk('/home/hiwonder/ActionSets'):
            print(root)
            print(dirs)
            print(files)
            return ActionSetListResponse(action_sets=files)

    def save_actionset_cb(self, req):
        with open(os.path.join('/home/hiwonder/ActionSets', req.file_name), 'w') as f:
            print(req.data)
            f.write(req.data)
        return [True, '']

    def read_actionset_cb(self, req):
        try:
            with open(os.path.join('/home/hiwonder/ActionSets', req.file_name)) as f:
                data = json.load(f)
                return [True, json.dumps(data)]
        except Exception as e:
            return [False, str(e)]

    def remove_actionset_cb(self, req):
        os.remove(os.path.join('/home/hiwonder/ActionSets', req.file_name))
        return [True, '']

    def run_online(self, msg):
        repeat = msg.repeat
        action_set = json.loads(msg.data)
        success = True
        count = 0
        print(action_set)
        for i in range(repeat):
            if success:
                for action in action_set['data']:
                    duration = action['duration']
                    position = action['position']
                    pwm1, pwm2 = action['pwm_servos']
                    sucker = True if action['sucker'] else False
                    self._jetmax.set_position(position, duration)
                    hiwonder.pwm_servo1.set_position(pwm1, duration)
                    hiwonder.pwm_servo2.set_position(pwm2, duration)
                    if self._sucker.get_state() != sucker:
                        self._sucker.set_state(sucker)
                    rospy.sleep(duration)
                    if self.action_set_online.is_preempt_requested():
                        rospy.loginfo('run online: Preempted')
                        self.action_set_online.set_preempted()
                        success = False
                        break
                    self.action_set_online.publish_feedback(ActionSetRawFeedback(index=i + 1, count=count))
            count += 1
        if success:
            self.action_set_online.set_succeeded(ActionSetRawResult(True))

    def _start_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def run_action_set_a(self, action_set, repeat):
        for i in range(repeat):
            for action in action_set['data']:
                duration = action['duration']
                position = action['position']
                pwm1, pwm2 = action['pwm_servos']
                sucker = action['sucker']
                self._jetmax.set_position(position, duration)
                hiwonder.pwm_servo1.set_position(pwm1, duration)
                hiwonder.pwm_servo2.set_position(pwm2, duration)
                self._sucker.set_state(True if sucker != 0 else False)
                await asyncio.sleep(duration)

    def run_action_set(self, action_set, repeat, block=False, callback=None):
        with self._lock:
            self.stop_action_set()
            self._running = asyncio.run_coroutine_threadsafe(self.run_action_set_a(action_set, repeat), self._loop)
            if block:
                self._running.result()
                self._running = None
            else:
                if callable(callback):
                    self._loop.call_soon_threadsafe(self._running.add_done_callback, callback)
                    return self._running

    def run_action_set_str(self, action_set_str, repeat, block):
        action_set = json.loads(action_set_str)
        self.run_action_set(action_set, repeat, block)

    def run_action_set_file(self, path, repeat, block):
        with open(path) as f:
            action_set = json.load(f)
        self.run_action_set(action_set, repeat, block)

    def stop_action_set(self):
        with self._lock:
            if self._running:
                self._loop.call_soon_threadsafe(self._running.cancel)
                self._running = None
