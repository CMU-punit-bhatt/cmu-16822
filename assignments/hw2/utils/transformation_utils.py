import logging
import numpy as np

def get_projection_matrix_from_correspondences(points_2d, points_3d):
    assert points_2d.shape[0] >= 6
    assert points_3d.shape[0] >= 6
    assert points_2d.shape[0] == points_3d.shape[0]

    n = points_2d.shape[0]

    if points_2d.shape[1] == 2:
        points_2d = np.hstack((points_2d, np.ones((n, 1))))
    else:
        points_2d /= points_2d[..., -1].reshape(-1, 1)

    if points_3d.shape[1] == 3:
        points_3d = np.hstack((points_3d, np.ones((n, 1))))
    else:
        points_3d /= points_3d[..., -1].reshape(-1, 1)

    # Constructing matrix A (n * 2, 12)
    A = np.zeros((n * 2, 12))

    for i in range(n):
        x, y, _ = points_2d[i]
        X, Y, Z, _ = points_3d[i]
        A[i * 2: (i + 1) * 2] = [
            [X, Y, Z, 1, 0, 0, 0, 0, - x * X, - x * Y, - x * Z, - x],
            [0, 0, 0, 0, X, Y, Z, 1, - y * X, - y * Y, - y * Z, - y]
        ]

    _, _, v_t = np.linalg.svd(A)
    p = v_t[-1]
    p /= p[-1]

    return p.reshape(3, 4)

def get_vanishing_points(annotations):
    assert annotations.shape[1:] == (2, 2, 2)

    n_pairs = annotations.shape[0]

    # Getting homogenous points - (n_pairs, 2, 3) each
    points1 = np.concatenate(
        (annotations[:, :, 0], np.ones((n_pairs, 2, 1))),
        axis=-1
    )
    points2 = np.concatenate(
        (annotations[:, :, 1], np.ones((n_pairs, 2, 1))),
        axis=-1
    )

    # Getting n pairs of parallel lines - (n_pairs, 2, 3)
    parallel_lines = np.cross(points1, points2)
    parallel_lines /= parallel_lines[:, :, -1].reshape(n_pairs, 2, 1)

    # Getting points of intersection for each pair of parallel lines - (n_pairs, 3)
    points_intersection = np.cross(parallel_lines[:, 0], parallel_lines[:, 1])
    points_intersection /= points_intersection[:, -1].reshape(-1, 1)

    assert points_intersection.shape == (n_pairs, 3)

    return points_intersection


def get_intrinsic_matrix_from_vanishing_points(vanishing_points):

    assert vanishing_points.shape == (3, 3)

    n_points = vanishing_points.shape[0]

    A = []

    for i in range(n_points):
        for j in range(i + 1, n_points):
            xi, yi, zi = vanishing_points[i]
            xj, yj, zj = vanishing_points[j]
            A.append([
                xi * xj + yi * yj,
                xi * zj + zi * xj,
                yi * zj + zi * yj,
                zi * zj
            ])

    A = np.asanyarray(A)

    _, _, v_t = np.linalg.svd(A)
    w1, w2, w3, w4 = v_t[-1]
    omega = np.array([
        [w1, 0, w2],
        [0, w1, w3],
        [w2, w3, w4]
    ])

    return omega_to_K(omega)

def get_intrinsic_matrix_from_homographies(homographies):
    assert len(homographies) == 3

    A = []

    get_coefficients = lambda h_ht: [
        h_ht[0, 0], h_ht[0, 1] + h_ht[1, 0], h_ht[0, 2] + h_ht[2, 0],
        h_ht[1, 1], h_ht[1, 2] + h_ht[2, 1], h_ht[2, 2]
    ]

    for H in homographies:
        h1, h2 = H[:, 0].reshape(-1, 1), H[:, 1].reshape(-1, 1)
        h_ht1 = np.matmul(h1, h2.T)
        h_ht2 = np.matmul(h1, h1.T) - np.matmul(h2, h2.T)
        A.append(get_coefficients(h_ht1))
        A.append(get_coefficients(h_ht2))

    A = np.asanyarray(A)

    assert A.shape == (6, 6) # Nice

    _, _, v_t = np.linalg.svd(A)
    w = v_t[-1]
    omega = np.array([
        [w[0], w[1], w[2]],
        [w[1], w[3], w[4]],
        [w[2], w[4], w[5]]
    ])

    return omega_to_K(omega)

def get_homography_matrix(orig_points, persp_points):
    assert orig_points.shape == persp_points.shape

    # Constructing matrix A
    A = np.zeros((8, 9))

    for i in range(4):
        xs, ys, _ = orig_points[i]
        xd, yd, _ = persp_points[i]
        A[i * 2: (i + 1) * 2] = [
            [xs, ys, 1, 0, 0, 0, - xs * xd, - ys * xd, - xd],
            [0, 0, 0, xs, ys, 1, - xs * yd, - ys * yd, - yd]
        ]

    _, _, v_t = np.linalg.svd(A)
    H = v_t[-1].reshape(3, 3)
    H /= H[-1, -1]

    return H


def homogenize(points):
    assert len(points.shape) == 2

    return np.hstack((points, np.ones((points.shape[0], 1))))

def get_vector_normal_to_plane(annotations, K):
    # 4 points annotated.
    assert annotations.shape == (4, 3)

    # Assuming the order of annotations is clockwise or counter-clockwise.
    annotations_vp1 = annotations.reshape(2, 2, 3)[..., :2]
    annotations_vp2 = np.roll(annotations, -1, axis=0).reshape(2, 2, 3)[..., :2]

    vanishing_points = get_vanishing_points(
        np.stack((annotations_vp1, annotations_vp2), axis=0)
    ).reshape(-1, 3, 1)

    assert vanishing_points.shape == (2, 3, 1)

    direction_vectors = np.matmul(np.linalg.inv(K), vanishing_points)

    assert direction_vectors.shape == (2, 3, 1)

    n = np.cross(
        direction_vectors[0].reshape(-1),
        direction_vectors[1].reshape(-1)
    )

    n /= np.linalg.norm(n)

    assert np.isclose(np.linalg.norm(n), 1, rtol=1e-05, atol=1e-08)

    return n

def get_plane_equation(annotations, K, known_point_2d, depth):
    normal = get_vector_normal_to_plane(annotations, K)

    logging.info(f'Plane normal - {normal}')

    assert normal.shape == (3,)
    assert known_point_2d.shape == (3,)

    # Back proj - X = P_inv x
    known_point_3d = np.linalg.inv(K) @ known_point_2d.reshape(-1, 1)

    # lambda l
    l = depth / np.linalg.norm(known_point_3d)
    logging.info(f'Known point lambda - {l}')

    # Equation of plane - nX + a = 0
    a = - np.dot(normal, known_point_3d) * l

    return np.array([*normal, *a])


def omega_to_K(omega):
    assert omega.shape == (3, 3)

    L = np.linalg.cholesky(omega)
    K = np.linalg.inv(L.T)
    K /= K[-1, -1]

    return K
