import numpy as np


def normalize_pts(pts, im_size):
    """Normalize image coordinate using the image size.

    Pre-processing of correspondences before passing them to the network to be 
    independent of image resolution.
    Re-scales points such that max image dimension goes from -0.5 to 0.5.
    In-place operation.

    Keyword arguments:
    pts -- 3-dim array conainting x and y coordinates in the last dimension, 
    first dimension should have size 1.
    im_size -- image height and width
    """

    pts[0, :, 0] -= float(im_size[1]) / 2
    pts[0, :, 1] -= float(im_size[0]) / 2
    pts /= float(max(im_size))


def denormalize_pts(pts, im_size):
    """Undo image coordinate normalization using the image size.

    In-place operation.

    Keyword arguments:
    pts -- N-dim array conainting x and y coordinates in the first dimension
    im_size -- image height and width
    """
    pts *= max(im_size)
    pts[0] += im_size[1] / 2
    pts[1] += im_size[0] / 2


def translationError(pose_error):
    return np.sqrt(pose_error[0, 2]**2 + pose_error[1, 2]**2)


def get_inverse_tf(T):
    T2 = np.identity(3)
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    t = np.reshape(t, (2, 1))
    T2[0:2, 0:2] = R.transpose()
    t = np.matmul(-1 * R.transpose(), t)
    T2[0, 2] = t[0]
    T2[1, 2] = t[1]
    return T2


def enforce_orthogonality(R):
    epsilon = 0.001
    if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
        print("ERROR: this is not a proper rigid transformation!")
    a = (R[0, 0] + R[1, 1]) / 2
    b = (-R[1, 0] + R[0, 1]) / 2
    sum = np.sqrt(a**2 + b**2)
    a /= sum
    b /= sum
    R[0, 0] = a
    R[0, 1] = b
    R[1, 0] = -b
    R[1, 1] = a


def get_transform(x, y, theta):
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    if np.linalg.det(R) != 1.0:
        enforce_orthogonality(R)
    T = np.identity(3)
    T[0:2, 0:2] = R
    T[0, 2] = x
    T[1, 2] = y
    return T
