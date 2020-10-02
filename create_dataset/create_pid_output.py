"""
Simulate random input for drone and target position, velocity, etc.
And then generate the corresponding thrust output dictated by the old drone PID implementation.
"""
import os
import sys

sys.path.append(os.getcwd())
import numpy as np
from controller.physical_model import QuadcopterPhysics
from create_dataset.util import *
from controller.onboard_computer import PIDControlUNnit
from controller.pid import PID
from controller.choose_paramset import get_paramset, convert_paramset_2_float

# from controller.geometry import *
from controller.parameters import *
from controller import setup_pids

delta_t = 0.01


def status_update(position, velocity, angle, angle_vel, thrust, time, lin_targets=None):
    """
    Print a formatted string with current velocity, position, etc.
    """
    print(f"time: {time}")

    pos_title = "position:".ljust(25)
    vel_title = "velocity:".rjust(25)
    print(pos_title + vel_title)
    for i in range(3):
        print(f"{position[i, 0]}".ljust(25) + f"{velocity[i, 0]}".rjust(25))
    print()
    angle_title = "angle:".ljust(25)
    angle_vel_title = "angular velocity:".rjust(25)
    print(angle_title + angle_vel_title)
    for i in range(3):
        print(f"{angle[i, 0]}".ljust(25) + f"{angle_vel[i, 0]}".rjust(25))

    print("\nthrust:")
    thrust_string = ""
    for i in range(4):
        thrust_string += f"{thrust[i]}\t"
    print(thrust_string)
    print()

    if lin_targets is not None:
        print(f"x_target: {lin_targets[-3]}")
        print(f"y_target: {lin_targets[-2]}")
        print(f"z_target: {lin_targets[-1]}")
        print()


def generate_ground_truth(paramset_num="11"):
    lab_pos = set_random_drone_position(a=-10, b=10)
    lab_lin_vel = set_random_drone_velocity()
    drone_angle = set_random_drone_angle()
    drone_angle_vel = set_random_drone_angular_velocity()
    coin_position = set_random_coin_position(-10, 10)

    pid_thrust = compute_pid_output(
        paramset_num, coin_position, lab_pos, lab_lin_vel, drone_angle, drone_angle_vel
    )

    return (
        np.concatenate(
            (coin_position, lab_pos, lab_lin_vel, drone_angle, drone_angle_vel)
        ),
        pid_thrust,
    )


def compute_pid_output(
    paramset_num, coin_position, lab_pos, lab_lin_vel, drone_angle, drone_angle_vel
):
    controller, lin_pids, rot_pids = setup_pids(paramset_num, delta_t=0.01)

    lin_targets = coin_position
    rot_targets = np.zeros((3, 1))

    [rot_pids[i].set_setpoint(rot_targets[i, 0]) for i in range(3)]
    [lin_pids[i].set_setpoint(lin_targets[i, 0]) for i in range(3)]

    pid_thrust = controller.translate_input_to_thrust(
        lab_pos,
        lab_lin_vel,
        drone_angle,
        drone_angle_vel,
        drone_angle,
        lin_pids,
        rot_pids,
    )

    return pid_thrust


if __name__ == "__main__":
    paramset_num = "11"
    if len(sys.argv) > 1:
        paramset_num = paramset_num

    paramset = get_paramset(paramset_num)
    (
        kp_x,
        kp_y,
        kp_z,
        ki_x,
        ki_y,
        ki_z,
        kd_x,
        kd_y,
        kd_z,
        kp_rol,
        kp_pit,
        kp_yaw,
        ki_rol,
        ki_pit,
        ki_yaw,
        kd_rol,
        kd_pit,
        kd_yaw,
    ) = convert_paramset_2_float(paramset)

    lin_pids = [
        PID(
            kp=kp_x,
            ki=ki_x,
            kd=kd_x,
            timeStep=delta_t,
            setValue=0,
            integralRange=2,
            calculateFlag="velocity",
            outputLimitRange=[-np.pi / 4, np.pi / 4],
        ),
        PID(
            kp=kp_y,
            ki=ki_y,
            kd=kd_y,
            timeStep=delta_t,
            setValue=0,
            integralRange=2,
            calculateFlag="velocity",
            outputLimitRange=[-np.pi / 4, np.pi / 4],
        ),
        PID(
            kp=kp_z,
            ki=ki_y,
            kd=kd_z,
            timeStep=delta_t,
            setValue=0,
            integralRange=2,
            calculateFlag="velocity",
        ),
    ]

    rot_pids = [
        PID(
            kp=kp_rol,
            ki=ki_rol,
            kd=kd_rol,
            timeStep=delta_t,
            setValue=0 * np.pi / 180,
            integralRange=2,
            calculateFlag="velocity",
            outputLimitRange=[-np.pi / 4, np.pi / 4],
        ),
        PID(
            kp=kp_pit,
            ki=ki_pit,
            kd=kd_pit,
            timeStep=delta_t,
            setValue=0 * np.pi / 180,
            integralRange=2,
            calculateFlag="velocity",
            outputLimitRange=[-np.pi / 4, np.pi / 4],
        ),
        PID(
            kp=kp_yaw,
            ki=ki_yaw,
            kd=kd_yaw,
            timeStep=delta_t,
            setValue=0 * np.pi / 180,
            integralRange=2,
            calculateFlag="velocity",
            outputLimitRange=[-np.pi / 6, np.pi / 6],
        ),
    ]

    qc = QuadcopterPhysics(
        mass_center=mass_center,
        mass_motor=mass_motor,
        radius_motor_center=radius_motor_center,
        coef_force=coef_force,
        coef_moment=coef_moment,
        coef_wind=coef_wind,
        gravity=gravity,
        mass_payload=mass_payload,
        x_payload=x_payload,
        y_payload=y_payload,
        I_x=I_x,
        I_y=I_y,
        I_z=I_z,
    )
    controller = PIDControlUNnit(pids=[lin_pids, rot_pids], quadcopter=qc)

    lin_targets = set_random_coin_position(-10, 10)
    # for testing
    lin_targets = np.array([[0], [1], [10]])
    rot_targets = np.zeros((3, 1))
    [rot_pids[i].set_setpoint(rot_targets[i, 0]) for i in range(3)]
    [lin_pids[i].set_setpoint(lin_targets[i, 0]) for i in range(3)]

    wind_speed = np.zeros((3, 1))

    lab_pos = set_random_drone_position(a=-10, b=10)
    lab_lin_vel = set_random_drone_velocity()
    drone_angle = set_random_drone_angle()
    drone_angle_vel = set_random_drone_angular_velocity()

    # # for testing
    # lab_pos = np.array([[0], [0], [10]])
    # lab_lin_vel = np.array([[0], [0.5], [0]]) # NOTE velocity input doesn't seem to have any impact

    # drone_angle = np.array([[0], [0], [0]])
    # drone_angle_vel = np.zeros((3, 1))

    thrust = controller.translate_input_to_thrust(
        lab_pos,
        lab_lin_vel,
        drone_angle,
        drone_angle_vel,
        drone_angle,
        lin_pids,
        rot_pids,
    )

    status_update(
        lab_pos,
        lab_lin_vel,
        drone_angle,
        drone_angle_vel,
        thrust,
        0,
        lin_targets=lin_targets,
    )
    print(f"")

    # TODO
    # How exactly did PID know when to counter-steer?
    # Was it through some hidden inner state?
    # Otherwise, what should happen if:
    # drone position: (0, 0, 0)
    # target position:(1, 0, 0)
    # drone velocity: (20, 0, 0)

    # I would expect the PID to already tell the drone to counter-steer
