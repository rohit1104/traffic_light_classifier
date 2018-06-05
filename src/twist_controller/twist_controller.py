import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.throttle_controller = PID(0.1, 0.001, 3, 0, accel_limit)
        self.velocity_lpf = LowPassFilter(0.5, 0.02)
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.last_time = rospy.get_time()

    def control(self, reqd_linear_vel, reqd_angular_vel, curr_linear_vel, dbw_enabled):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        curr_linear_vel = self.velocity_lpf.filt(curr_linear_vel)

        steering = self.yaw_controller.get_steering(reqd_linear_vel, reqd_angular_vel, curr_linear_vel)

        cte_vel = reqd_linear_vel - curr_linear_vel
        current_time = rospy.get_time()
        time_diff = current_time - self.last_time
        self.last_time = current_time
        throttle = self.throttle_controller.step(cte_vel, time_diff)

        brake = 0
        if reqd_linear_vel == 0 and curr_linear_vel < 0.1:
            throttle = 0
            brake = 700
        elif throttle < 0.1 and cte_vel < 0:
            throttle = 0
            decel = max(cte_vel, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
