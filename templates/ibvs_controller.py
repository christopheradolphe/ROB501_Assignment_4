import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    """
    v = np.zeros((6, 1))

    # Determine the number of points
    n = pts_obs.shape[1]

    # Initialize empty Jacobian
    J = np.zeros((0,6))

    for i in range(n):
        # Compute Jacobian for each image plane point
        J_point = ibvs_jacobian(K,pts_obs[:,i],zs[i])
        J = np.vstack([J,J_point])
    
    # Compute the pseduoinverse of J
    J_pseudo = np.linalg.pinv(J)

    # Compute error of image points
    error = (pts_des - pts_obs).reshape(-1,1)
    
    # Compute velocity using Corke 2023 (Equation 15.14)
    v = gain * J_pseudo @ error

    correct = isinstance(v, np.ndarray) and \
        v.dtype == np.float64 and v.shape == (6, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return v