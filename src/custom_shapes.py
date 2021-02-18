from pyrep.objects import Shape, Dummy, Object
from pyrep.robots.arms.arm import Arm
from pyrep.const import ObjectType


class StatefulObject(Dummy):
    def __init__(self, name_or_handle, pyrep):
        super().__init__(name_or_handle)
        self._pyrep = pyrep

    def get_state(self):
        funcname = "getState@{}".format(self.get_name())
        ints, floats, strings, bytes = self._pyrep.script_call(funcname, 1)
        if not ints:
            raise ValueError(
                "Script return value incorect ({})".format(self.get_name())
            )
        return ints[0]

    def set_state(self, on):
        funcname = "setState@{}".format(self.get_name())
        ints, floats, strings, bytes = self._pyrep.script_call(
            funcname,
            1,
            ints=[int(on)]
        )
        if not ints:
            raise ValueError(
                "Script return value incorect ({})".format(self.get_name())
            )
        return ints[0]

    def set_goal(self, on):
        funcname = "setGoal@{}".format(self.get_name())
        ints, floats, strings, bytes = self._pyrep.script_call(
            funcname,
            1,
            ints=[int(on)]
        )
        if not ints:
            raise ValueError(
                "Script return value incorect ({})".format(self.get_name())
            )
        return ints[0]


class TapShape(StatefulObject):
    def __init__(self, name_or_handle, pyrep):
        super().__init__(name_or_handle, pyrep)
        proximity_sensors = self.get_objects_in_tree(
            object_type=ObjectType.PROXIMITY_SENSOR
        )
        self.proximity_sensor_0 = next(
            s for s in proximity_sensors
            if s.get_name().startswith("proximity_sensor_0")
        )
        self.proximity_sensor_1 = next(
            s for s in proximity_sensors
            if s.get_name().startswith("proximity_sensor_1")
        )
        self.joint = self.get_objects_in_tree(
            object_type=ObjectType.JOINT
        )[0]


class ButtonShape(StatefulObject):
    def __init__(self, name_or_handle, pyrep):
        super().__init__(name_or_handle, pyrep)
        self.proximity_sensor = self.get_objects_in_tree(
            object_type=ObjectType.PROXIMITY_SENSOR
        )[0]


class LeverShape(StatefulObject):
    def __init__(self, name_or_handle, pyrep):
        super().__init__(name_or_handle, pyrep)
        proximity_sensors = self.get_objects_in_tree(
            object_type=ObjectType.PROXIMITY_SENSOR
        )
        self.proximity_sensor_0 = next(
            s for s in proximity_sensors
            if s.get_name().startswith("proximity_sensor_0")
        )
        self.proximity_sensor_1 = next(
            s for s in proximity_sensors
            if s.get_name().startswith("proximity_sensor_1")
        )
        self.joint = self.get_objects_in_tree(
            object_type=ObjectType.JOINT
        )[0]


class Kuka(Arm):
    def __init__(self, name_or_handle, count=0):
        if type(name_or_handle) is int:
            name = Object.get_object_name(name_or_handle)
            loc = name.find('#')
            if loc != -1:
                name = name[:loc]
        else:
            name = name_or_handle
        super().__init__(count=count, name=name, num_joints=7)
        # print([j.get_name() for j in self.joints])
