import logging
import numpy as np
from scipy.optimize import least_squares

from utils.utils import get_points_lines_dist

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

def get_skew_symmetric_matrix_from_vector(x):

    assert x.shape[-1] == 3

    # [0 -x3 x2]
    # [x3 0 -x1]
    # [-x2 x1 0]

    res_shape = (3, 3) if len(x.shape) == 1 else (x.shape[0], 3, 3)
    result_mat = np.zeros((res_shape))

    result_mat[..., 0, 1] = - x[..., 2]
    result_mat[..., 0, 2] = x[..., 1]
    result_mat[..., 1, 2] = - x[..., 0]
    result_mat[..., 1, 0] = x[..., 2]
    result_mat[..., 2, 0] = - x[..., 1]
    result_mat[..., 2, 1] = x[..., 0]

    return result_mat

def get_normalized_points(points):

    assert len(points.shape) == 2
    assert points.shape[-1] == 2

    point_0 = np.mean(points, axis=0)

    d_avg = np.mean(np.sqrt(np.sum((points - point_0) ** 2, axis=-1)))
    s = (2 ** 0.5) / d_avg

    T = np.array([
        [s, 0, - s * point_0[0]],
        [0, s, - s * point_0[1]],
        [0, 0, 1]
    ])

    norm_points = T @ homogenize(points).reshape(-1, 3, 1)

    return T, norm_points.reshape(-1, 3)

def get_F_8_points_alg(points1, points2):

    assert len(points1.shape) == 2 and len(points2.shape) == 2
    assert points1.shape[-1] == 2 and points2.shape[-1] == 2

    T1, norm_points1 = get_normalized_points(points1)
    T2, norm_points2 = get_normalized_points(points2)

    A = np.zeros((norm_points1.shape[0], 9))

    A[:, 0] = norm_points2[:, 0] * norm_points1[:, 0]
    A[:, 1] = norm_points2[:, 0] * norm_points1[:, 1]
    A[:, 2] = norm_points2[:, 0] * norm_points1[:, 2]
    A[:, 3] = norm_points2[:, 1] * norm_points1[:, 0]
    A[:, 4] = norm_points2[:, 1] * norm_points1[:, 1]
    A[:, 5] = norm_points2[:, 1] * norm_points1[:, 2]
    A[:, 6] = norm_points2[:, 2] * norm_points1[:, 0]
    A[:, 7] = norm_points2[:, 2] * norm_points1[:, 1]
    A[:, 8] = norm_points2[:, 2] * norm_points1[:, 2]

    _, _, v_t = np.linalg.svd(A)
    F = v_t[-1].reshape(3, 3)

    # Projecting F to rank 2
    u, d, v_t = np.linalg.svd(F)
    d[-1] = 0
    F = u @ (np.eye(3) * d) @ v_t

    # Transforming F back to pixel space.
    F = T2.T @ F @ T1

    return F / F[-1, -1]

def get_F_7_points_alg(points1, points2):

    assert len(points1.shape) == 2 and len(points2.shape) == 2
    assert points1.shape[-1] == 2 and points2.shape[-1] == 2

    T1, norm_points1 = get_normalized_points(points1)
    T2, norm_points2 = get_normalized_points(points2)

    A = np.zeros((norm_points1.shape[0], 9))

    A[:, 0] = norm_points2[:, 0] * norm_points1[:, 0]
    A[:, 1] = norm_points2[:, 0] * norm_points1[:, 1]
    A[:, 2] = norm_points2[:, 0] * norm_points1[:, 2]
    A[:, 3] = norm_points2[:, 1] * norm_points1[:, 0]
    A[:, 4] = norm_points2[:, 1] * norm_points1[:, 1]
    A[:, 5] = norm_points2[:, 1] * norm_points1[:, 2]
    A[:, 6] = norm_points2[:, 2] * norm_points1[:, 0]
    A[:, 7] = norm_points2[:, 2] * norm_points1[:, 1]
    A[:, 8] = norm_points2[:, 2] * norm_points1[:, 2]

    _, _, v_t = np.linalg.svd(A)
    F1 = v_t[-1].reshape(3, 3)
    F2 = v_t[-2].reshape(3, 3)

    # Projecting F to rank 2
    func = lambda l: np.linalg.det(l * F1 + (1 - l) * F2)

    c = np.zeros(4)

    c[3] = func(0)
    c[1] = (func(1) + func(-1)) / 2 - c[3]
    c[0] = (func(2) + c[3] - 2 * c[1] - 2 * func(1)) / 6
    c[2] = func(1) - c[0] - c[1] - c[3]

    roots = np.roots(c)
    roots = roots[np.isreal(roots)]

    F_list = [np.real(l) * F1 + (1 - np.real(l)) * F2 for l in roots]
    F_list = [T2.T @ F @ T1 for F in F_list]
    F_list = [F / F[-1, -1] for F in F_list]

    # Finding best F.
    min_err = None

    for i, F in enumerate(F_list):
        err = get_epipolar_error(homogenize(points1), homogenize(points2), F)
        err = err.sum()

        logging.info(
            f'Fundamental matrix candidate {i}: \n{F}\n'
        )
        logging.info(
            f'Candidate {i} squared dist error: \n{err}\n'
        )

        if min_err is None or min_err > err:
            min_err = err
            best_F_idx = i

    F = F_list[best_F_idx]

    return F

def get_epipolar_line(points, F):

    assert points.shape[-1] == 3

    lines = F @ points.reshape(-1, 3, 1)

    return lines.reshape(-1, 3)

def get_essential_matrix(F, K1, K2):

    assert F.shape == K1.shape == K2.shape == (3, 3)

    return K2.T @ F @ K1

def get_epipolar_error(points1, points2, F):
    assert points1.shape == points2.shape
    assert points1.shape[-1] == points2.shape[-1] == 3

    epipolar_lines1 = F @ points1.reshape(-1, 3, 1)
    # epipolar_lines1 = epipolar_lines1.reshape(-1, 3) / np.linalg.norm(epipolar_lines1, axis=-1)
    sqrd_dists1 = get_points_lines_dist(
        points2.reshape(-1, 3),
        epipolar_lines1.reshape(-1, 3)
    )

    epipolar_lines2 = F.T @ points2.reshape(-1, 3, 1)
    # epipolar_lines2 = epipolar_lines2.reshape(-1, 3) / np.linalg.norm(epipolar_lines2, axis=-1)
    sqrd_dists2 = get_points_lines_dist(
        points1.reshape(-1, 3),
        epipolar_lines2.reshape(-1, 3)
    )

    # all_errors = np.sqrt(np.concatenate((sqrd_dists1, sqrd_dists2)))
    all_errors = np.sqrt(sqrd_dists1) + np.sqrt(sqrd_dists2)
    assert all_errors.shape[0] == sqrd_dists1.shape[0]

    return all_errors.flatten()

def ransac_F(
    alg_func,
    error_func,
    points1,
    points2,
    n_iters,
    thresh,
    n_corrs
):

    assert points1.shape == points2.shape
    assert points1.shape[1] == points2.shape[1] == 2

    N = points1.shape[0]
    n_inliers_list = []
    best_inliers = None

    for i in range(n_iters):
        rand_n = np.random.randint(0, N, size=n_corrs)
        F = alg_func(points1[rand_n], points2[rand_n])

        err = error_func(homogenize(points1), homogenize(points2), F)

        # Find inliers
        curr_inliers = np.argwhere(err < thresh)

        if best_inliers is None or curr_inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = curr_inliers
            best_F = F

        n_inliers_list.append((i, best_inliers.shape[0]))

    return best_F, best_inliers.flatten(), np.array(n_inliers_list)

def triangulate(points1, points2, P1, P2):

    assert points1.shape == points2.shape
    assert points1.shape[-1] == points2.shape[-1] == 2
    assert P1.shape == P2.shape == (3, 4)

    points1_skew_symm = get_skew_symmetric_matrix_from_vector(
        homogenize(points1)
    )
    points2_skew_symm = get_skew_symmetric_matrix_from_vector(
        homogenize(points2)
    )

    A = np.concatenate(
        (points1_skew_symm @ P1, points2_skew_symm @ P2),
        axis=-2
    )

    _, _, v_t = np.linalg.svd(A)

    points_3d = v_t[..., -1, :]

    assert points_3d.shape == (points1.shape[0], 4)

    return points_3d / points_3d[:, -1].reshape(-1, 1)

def bundle_adjestment(points1, points2, P1_init, P2_init):

    assert points1.shape == points2.shape
    assert points1.shape[-1] == points2.shape[-1] == 2
    assert P1_init.shape == P2_init.shape == (3, 4)

    def get_residuals(x, p1, p2):

        assert x.shape == (2 * 3 * 4,)

        P1 = x[: 12].reshape(3, 4)
        P2 = x[12:].reshape(3, 4)

        points_3d = triangulate(p1, p2, P1, P2).reshape(-1, 4, 1)

        reproj_p1 = (P1 @ points_3d).reshape(-1, 3)
        reproj_p1 /= reproj_p1[:, -1].reshape(-1, 1)

        reproj_p2 = (P2 @ points_3d).reshape(-1, 3)
        reproj_p2 /= reproj_p2[:, -1].reshape(-1, 1)

        residuals_p1 = np.linalg.norm(p1 - reproj_p1[:, :2], axis=-1)
        residuals_p2 = np.linalg.norm(p2 - reproj_p2[:, :2], axis=-1)

        assert residuals_p1.shape == residuals_p2.shape == (p1.shape[0],)

        return np.concatenate((residuals_p1, residuals_p2))

    x_init = np.concatenate((P1_init.reshape(-1), P2_init.reshape(-1)))
    result = least_squares(get_residuals, x_init, args=(points1, points2))

    P1_final = result.x[: 12].reshape(3, 4)
    P2_final = result.x[12:].reshape(3, 4)

    return P1_final, P2_final, result.cost