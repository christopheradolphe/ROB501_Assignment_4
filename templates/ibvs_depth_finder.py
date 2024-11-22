import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)

    # Find focal length and principal points from calibration matrix
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]

    # Extract linear and rotational velocities of last command
    v = v_cam[0:3]
    w = v_cam[3:]

    for i in range(n):
        # Calculate pixel coordiantes relative to principal point
        u_bar = pts_obs[0,i] - cx
        v_bar = pts_obs[1,i] - cy

        # Compute Jacobian for each image plane point
        J_t = np.array([
            [-f, 0, u_bar],
            [0, -f, v_bar]
        ])

        J_w =np.array([
            [(u_bar*v_bar)/ f, -(f**2 + u_bar**2)/f, v_bar],
            [(f**2+v_bar**2)/f, -(u_bar*v_bar)/f, -u_bar]
        ])
    
        # Compute A: J_t*v
        A = J_t @ v
        A = A.reshape(2,1)

        # Compute b: optical_flow - J_w * w
        optical_flow = (pts_obs[:,i] - pts_prev[:,i]).reshape(-1, order='F')
        optical_flow = optical_flow.reshape(2,1)
        b = optical_flow - J_w @ w
        b = b.reshape(2,1)

        # Solve for theta using linear lease squares
        theta = np.linalg.lstsq(A,b,rcond=None)[0]

        # Z estimate is inverse of thetas
        zs_est[i] = 1/theta


    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est