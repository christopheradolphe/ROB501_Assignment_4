import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    # Extract focal and principal points from calibration matrix
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]

    # Extract image plane point coordinates
    u, v = pt

    # Calculate pixel coordiantes relative to principal point
    u_bar = u - cx
    v_bar = v - cy

    # Initialize Jacobian to be all zeros
    J = np.zeros((2,6))

    # Populate Jacobian with non-zero values as per 2023 Corke Text (Equation 15.6)
    J[0,0] = -f/z
    J[0,2] = u_bar/z
    J[0,3] = u_bar*v_bar / f
    J[0,4] = -(f**2 + u_bar**2)/f
    J[0,5] = v_bar

    J[1,1] = -f/z
    J[1,2] = v_bar/z
    J[1,3] = (f**2+v_bar**2)/f
    J[1,4] = -(u_bar*v_bar)/f
    J[1,5] = -u_bar


    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J