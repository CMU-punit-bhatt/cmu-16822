from __future__ import annotations
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import draw_line, get_image_annotations, plot_images

def get_rectified_lines(l, H):
    return np.matmul(np.linalg.inv(H.T), l.reshape(-1, 3, 1))

def get_rectified_points(pts, H):
    return np.matmul(H, pts.reshape(-1, 3, 1))

def get_similarity_rectification_H(annotations):
    # Getting homogenous points - (2, 2, 3) each
    points1 = np.concatenate(
        (annotations[:, :, 0], np.ones((2, 2, 1))),
        axis=-1
    )
    points2 = np.concatenate(
        (annotations[:, :, 1], np.ones((2, 2, 1))),
        axis=-1
    )

    # Getting n pairs of perpendicular lines - (2, 2, 3)
    orthogonal_lines = np.cross(points1, points2)
    orthogonal_lines /= orthogonal_lines[:, :, -1].reshape(2, 2, 1)

    # Constructing L - (2, 2)
    L = np.zeros((2, 2))
    L[:, 0] = orthogonal_lines[:, 0, 0] * orthogonal_lines[:, 1, 0]
    L[:, 1] = orthogonal_lines[:, 0, 0] * orthogonal_lines[:, 1, 1] + \
        orthogonal_lines[:, 1, 0] * orthogonal_lines[:, 0, 1]

    # Constructing b - (2)
    b = - orthogonal_lines[:, 0, 1] * orthogonal_lines[:, 1, 1]

    # Solving Lx = b
    s = np.linalg.solve(L, b)
    S = np.array([[s[0], s[1]],
                  [s[1], 1]])

    # Constructing H_a
    u, d, u_t = np.linalg.svd(S)
    A = u @ (np.eye(2) * np.sqrt(d)) @ u_t
    H_a = np.eye(3)
    H_a[:2, :2] = A

    rectification_H = np.linalg.inv(H_a)

    return rectification_H

def get_affine_rectification_H(annotations):
    # Getting homogenous points - (2, 2, 3) each
    points1 = np.concatenate((annotations[:, :, 0], np.ones((2, 2, 1))), axis=-1)
    points2 = np.concatenate((annotations[:, :, 1], np.ones((2, 2, 1))), axis=-1)

    # Getting 2 pairs of parallel lines - (2, 2, 3)
    parallel_lines = np.cross(points1, points2)
    parallel_lines /= parallel_lines[:, :, -1].reshape(2, 2, 1)

    # Getting points of intersection for each pair of parallel lines - (2, 3)
    points_intersection = np.cross(parallel_lines[:, 0], parallel_lines[:, 1])
    points_intersection /= points_intersection[:, -1].reshape(-1, 1)

    # Getting imaged line at infinity - (3,)
    imaged_l_inf = np.cross(points_intersection[0], points_intersection[1])

    # Homogenizing imaged l infinity.
    assert imaged_l_inf[-1] != 0
    imaged_l_inf /= imaged_l_inf[-1]

    # Constructing H
    rectification_H = np.array(
        [[1, 0, 0],
        [0, 1, 0],
        imaged_l_inf]
    )

    return rectification_H

def calculate_test_lines_angle(
    img,
    rectification_H,
    colors,
    n_pairs=2,
    out_path=None
):

    assert len(colors) == n_pairs
    annotations, img_annotated = get_image_annotations(
        img,
        colors,
        n_pairs,
        title='Test Lines'
    )

    # Getting annotations - (n_pairs, 2, 2, 2).
    annotations = annotations.reshape(n_pairs, 2, 2, 2)

    # Getting homogenous points - (n_pairs, 2, 3) each
    points1 = np.concatenate(
        (annotations[:, :, 0], np.ones((n_pairs, 2, 1))),
        axis=-1
    )
    points2 = np.concatenate(
        (annotations[:, :, 1], np.ones((n_pairs, 2, 1))),
        axis=-1
    )

    # Getting 2 pairs of lines - (n_pairs, 2, 3)
    lines = np.cross(points1, points2)
    lines /= lines[:, :, -1].reshape(n_pairs, 2, 1)

    rectified_lines = get_rectified_lines(
        lines,
        rectification_H
    ).reshape(n_pairs, 2, 3)


    for i in range(n_pairs):
        logging.info(
            f'Cosine between original lines pair {i + 1} - ' +
            f'{cosine(*lines[i])}'
        )
        logging.info(
            f'Cosine between rectified lines pair {i + 1} - ' +
            f'{cosine(*rectified_lines[i])}'
        )

    rectified_img_annotated, rectification_Ht = warp_image(img, rectification_H)

    rectified_points1 = get_rectified_points(
        points1,
        rectification_Ht @ rectification_H
    ).reshape(-1, 3)
    rectified_points2 = get_rectified_points(
        points2,
        rectification_Ht @ rectification_H
    ).reshape(-1, 3)

    rectified_points1 /= rectified_points1[..., 2].reshape(-1, 1)
    rectified_points2 /= rectified_points2[..., 2].reshape(-1, 1)

    rectified_points1 = rectified_points1[..., :2].astype(np.int32).reshape(-1, 2)
    rectified_points2 = rectified_points2[..., :2].astype(np.int32).reshape(-1, 2)

    for i, pts in enumerate(zip(rectified_points1, rectified_points2)):
        rectified_img_annotated = draw_line(
            rectified_img_annotated,
            pts,
            colors[int(i / 2)]
        )

    plot_images(
        [img_annotated, rectified_img_annotated],
        ['Test Lines', 'Test Lines - Rectified'],
        (1, 2),
        out_path=out_path
    )



def normalize(v):
    return v / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def warp_image(img, H):
    h, w = img.shape[:2]
    pts = np.array([[0,0],[0,h],[w,h],[w,0]], dtype=np.float64).reshape(-1,1,2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)

    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    return result, Ht

def get_composite_img(orig_img, H, persp_img):
    warped_img = cv2.warpPerspective(
        orig_img,
        H,
        (persp_img.shape[1], persp_img.shape[0])
    )
    bg = np.amax(warped_img, axis=-1, keepdims=True)

    return np.uint8(bg == 0) * persp_img + warped_img


def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (
        np.sqrt(u[0]**2 + u[1]**2) * np.sqrt(v[0]**2 + v[1]**2)
    )