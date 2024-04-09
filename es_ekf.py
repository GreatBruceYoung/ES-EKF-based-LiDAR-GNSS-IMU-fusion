from rotations import Quaternion
import numpy as np
from numpy import matmul


def measurement_update(h_jac,sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    """Error-state extended Kalman Filter"""
    # Compute Kalman Gain
    # K_k = P_k * H_k.T * inv( H_k * P_k * H_k.T + R_k )
    try:
        temp = matmul(h_jac, matmul(p_cov_check, h_jac.T)) + sensor_var * np.eye(3)
        inv = np.linalg.inv(temp)
        # print("temp: ", temp.shape, "sensor_var: ", sensor_var)
        K = matmul(p_cov_check,
                   matmul(h_jac.T, inv))  # np.linalg.inv(matmul(h_jac, matmul(p_cov_check, h_jac.T)) + sensor_var )))

    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            raise "A singular matrix "

    # Compute error state
    # print("y_k size: ", y_k.shape, "h_jac size: ", h_jac.shape, "p_check size: ", p_check.shape, "P_CHECK: ", p_check)
    error_state = y_k - p_check  # matmul(h_jac[:3, :3], p_check)

    # Correct predicted state
    p_hat = p_check + matmul(K, error_state)[:3]
    v_hat = v_check + matmul(K, error_state)[3:6]
    # print("error_state ", error_state.shape, "K: ", K.shape, "q_check: ", q_check.shape)
    q_hat = Quaternion(axis_angle=matmul(K, error_state)[6:]).quat_mult_right(q_check)

    # Compute corrected covariance
    p_cov_hat = matmul(np.eye(9) - matmul(K, h_jac), p_cov_check)

    return p_hat, v_hat, q_hat, p_cov_hat