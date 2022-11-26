import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

def get_file_name(file, ext='.png'):
    return file.split('/')[-1].split('.')[0] + ext

def draw_line(img, pts, color, thickness=3):
    cv2.line(
        img,
        [int(i) for i in pts[0]],
        [int(i) for i in pts[1]],
        color,
        thickness=thickness
    )

    return img

def draw_line_eq(img, lines, colors, thickness=3):

    img_annotated = np.copy(img)

    for line, color in zip(lines, colors):
        assert lines.shape[-1] == 3

        _, w, _ = img.shape
        a, b, c = line.flatten()

        pt0 = (0, int(- c / b))
        pt1 = (w - 1, int(- (c + a * (w - 1)) / b))

        img_annotated = draw_line(img_annotated, [pt0, pt1], color, thickness)

    return img_annotated

def draw_point(img, pt, color, radius=10):
    cv2.circle(
        img,
        [int(i) for i in pt],
        radius=radius,
        color=color,
        thickness=cv2.FILLED
    )

    return img

def draw_points(img, points, colors, radius=10):
    for pt, color in zip(points, colors):
        img = draw_point(img, pt, color, radius)

    return img

def get_image_annotations(img, colors, n_pairs, title):
    assert len(colors) == n_pairs

    annotations = []
    img_annotated = np.copy(img)

    plt.title(title)

    for i in range(n_pairs):
        plt.imshow(img_annotated)
        points1 = np.array(plt.ginput(2, timeout=-1), dtype=np.int32)
        img_annotated = draw_line(img_annotated, points1, colors[i])
        plt.imshow(img_annotated)
        points2 = np.array(plt.ginput(2, timeout=-1), dtype=np.int32)
        img_annotated = draw_line(img_annotated, points2, colors[i])
        annotations.extend([points1.reshape(2, 2), points2.reshape(2, 2)])
    plt.close()

    annotations = np.array(annotations)

    return annotations, img_annotated

def get_point_annotations(img, colors, n_points):
    assert len(colors) == n_points

    annotations = []
    img_annotated = np.copy(img)

    for i in range(n_points):
        plt.imshow(img_annotated)
        points = np.array(plt.ginput(1, timeout=-1), dtype=np.int32)[0]
        img_annotated = draw_point(img_annotated, points, colors[i])
        annotations.append(points)
    plt.close()

    annotations = np.array(annotations)

    return annotations, img_annotated

def plot_images(images, titles, size, out_path=None):

    assert len(size) == 2
    assert len(titles) == size[0] * size[1]
    assert len(images) == size[0] * size[1]

    r, c = size

    _, axes = plt.subplots(*size, figsize=(12, 12))

    for i in range(r):
        for j in range(c):
            if r == 1 or c == 1:
                ax = axes[i + j]
            else:
                ax = axes[i, j]
            ax.set_title(titles[i * c + j])
            ax.imshow(images[i * c + j])

    if out_path is not None:
        plt.savefig(out_path)
    plt.show()

def read_image(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

def read_file(file):
    ext = file.split('.')[-1]

    if ext in ['jpg', 'jpeg', 'png']:
        return read_image(file)
    elif ext == 'txt':
        return np.loadtxt(file)
    elif ext == 'npy':
        return np.load(file, allow_pickle=True)
    else:
        return np.load(file)

def draw_correspondences(img1, img2, points1, points2, color=[0, 255, 0]):

    assert img1.shape[0] == img2.shape[0]
    assert points1.shape[-1] == points2.shape[-1] == 2
    assert points1.shape == points2.shape

    combined_img = np.hstack((img1, img2))
    new_points2 = points2 + [img1.shape[1], 0] # Moving the x coordinate accordingly for img2.

    for pt1, pt2 in zip(points1, new_points2):
        combined_img = draw_line(combined_img, [pt1, pt2], color, thickness=1)

    return combined_img

def get_points_lines_dist(points, lines):

    assert points.shape[-1] == lines.shape[-1] == 3
    assert points.shape[0] == lines.shape[0]

    sqrd_dists = np.square(
        np.sum(points * lines, axis=-1) / np.linalg.norm(lines[:, :2], axis=-1)
    )

    return sqrd_dists

def get_sift_features(img):

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Applying SIFT detector
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def compute_correspondences(img1, img2):

    keypoints1, descriptors1 = get_sift_features(img1)
    keypoints2, descriptors2 = get_sift_features(img2)

    keypoints1 = np.array([kp.pt for kp in keypoints1])
    keypoints2 = np.array([kp.pt for kp in keypoints2])

    matches = []

    for i in range(len(descriptors1)):
        des = descriptors1[i].reshape(1, -1)
        dists = np.linalg.norm(descriptors2 - des, axis=-1)

        best_match_idx = np.argsort(dists)[0]

        matches.append((i, best_match_idx))

    for i in range(len(descriptors2)):
        des = descriptors2[i].reshape(1, -1)
        dists = np.linalg.norm(descriptors1 - des, axis=-1)

        best_match_idx = np.argsort(dists)[0]

        matches.append((best_match_idx, i))

    matches = np.array(matches)

    return keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]