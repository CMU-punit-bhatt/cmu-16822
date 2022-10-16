import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_file_name(file, ext='.png'):
    return file.split('/')[-1].split('.')[0] + ext

def draw_line(img, pts, color, thickness=3):
    cv2.line(img, pts[0], pts[1], color, thickness=thickness)

    return img

def draw_point(img, pt, color, radius=10):
    cv2.circle(img, pt, radius=radius, color=color, thickness=cv2.FILLED)

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
    else:
        return np.load(file)
