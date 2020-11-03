from controller.physical_model import QuadcopterPhysics
from controller.onboard_computer import PIDControlUNnit
from controller.pid import PID
from controller.choose_paramset import get_paramset, convert_paramset_2_float

# from controller.geometry import *
from controller.parameters import *


def setup_pids(paramset_num, delta_t=0.01):
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
            calculateFlag="noFlush",
            outputLimitRange=[-np.pi / 4, np.pi / 4],
        ),
        PID(
            kp=kp_y,
            ki=ki_y,
            kd=kd_y,
            timeStep=delta_t,
            setValue=0,
            integralRange=2,
            calculateFlag="noFlush",
            outputLimitRange=[-np.pi / 4, np.pi / 4],
        ),
        PID(
            kp=kp_z,
            ki=ki_y,
            kd=kd_z,
            timeStep=delta_t,
            setValue=0,
            integralRange=2,
            calculateFlag="noFlush",
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

    return controller, lin_pids, rot_pids
