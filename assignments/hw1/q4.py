import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import(
    calculate_test_lines_angle,
    get_rectified_points,
    warp_image,
)
from utils.utils import (
    draw_line,
    get_file_name,
    get_image_annotations,
    plot_images,
    read_image
)

def direct_rectify_to_similarity(file, colors, out_path, load_annotations=False):

    img = read_image(file)

    if load_annotations:
        annotations, img_annotated = np.load(
            os.path.join(
                out_path, 'annotations_' + get_file_name(file, ext='.npy')
            ),
            allow_pickle=True
        )
    else:
        annotations, img_annotated = get_image_annotations(
            img,
            colors,
            n_pairs=5,
            title='Mark 5 Pairs of Orthogonal Lines'
        )

        # Getting annotations - (5, 2, 2, 2).
        annotations = annotations.reshape(5, 2, 2, 2)

        np.save(
            os.path.join(
                out_path, 'annotations_' + get_file_name(file, ext='.npy')
            ),
            (annotations, img_annotated)
        )

    # Getting homogenous points - (5, 2, 3) each
    points1 = np.concatenate(
        (annotations[:, :, 0], np.ones((5, 2, 1))),
        axis=-1
    )
    points2 = np.concatenate(
        (annotations[:, :, 1], np.ones((5, 2, 1))),
        axis=-1
    )

    # Getting n pairs of perpendicular lines - (5, 2, 3)
    orthogonal_lines = np.cross(points1, points2)
    orthogonal_lines /= orthogonal_lines[:, :, -1].reshape(5, 2, 1)

    # Constructing L - (5, 6)
    L = np.zeros((5, 6))
    L[:, 0] = orthogonal_lines[:, 0, 0] * orthogonal_lines[:, 1, 0]
    L[:, 1] = (
        orthogonal_lines[:, 0, 0] * orthogonal_lines[:, 1, 1] +
        orthogonal_lines[:, 0, 1] * orthogonal_lines[:, 1, 0]
    ) / 2
    L[:, 2] = orthogonal_lines[:, 0, 1] * orthogonal_lines[:, 1, 1]
    L[:, 3] = (
        orthogonal_lines[:, 0, 0] * orthogonal_lines[:, 1, 2] +
        orthogonal_lines[:, 0, 2] * orthogonal_lines[:, 1, 0]
    ) / 2
    L[:, 4] = (
        orthogonal_lines[:, 0, 1] * orthogonal_lines[:, 1, 2] +
        orthogonal_lines[:, 0, 2] * orthogonal_lines[:, 1, 1]
    ) / 2
    L[:, 5] = orthogonal_lines[:, 0, 2] * orthogonal_lines[:, 1, 2]

    _, _, vt = np.linalg.svd(L)
    c = vt[-1]

    # Constructing C prime star infinity.
    C_inf = np.array(
        [[c[0],     c[1] / 2, c[3] / 2],
        [c[1] / 2, c[2]    , c[4] / 2],
        [c[3] / 2, c[4] / 2, c[5]]]
    )

    # Check notes for this derivation if ever in doubt :)
    u, s, _ = np.linalg.svd(C_inf)

    rectification_H = np.eye(3)
    rectification_H[0, 0] = np.sqrt(s[0])
    rectification_H[1, 1] = np.sqrt(s[1])

    rectification_H = np.linalg.inv(np.matmul(u, rectification_H))
    rectification_H /= rectification_H[-1, -1]

    rectified_img, rectification_Ht = warp_image(img, rectification_H)

    calculate_test_lines_angle(
        img,
        rectification_H,
        colors[:3],
        n_pairs=3,
        out_path=os.path.join(out_path, 'test_lines_' + get_file_name(file))
    )


    # Getting perpendicular annotations on final similarity rectified image.
    rectified_points1 = get_rectified_points(
        points1,
        rectification_Ht.dot(rectification_H)
    ).reshape(-1, 3)
    rectified_points2 = get_rectified_points(
        points2,
        rectification_Ht.dot(rectification_H)
    ).reshape(-1, 3)
    rectified_points1 /= rectified_points1[..., 2].reshape(-1, 1)
    rectified_points2 /= rectified_points2[..., 2].reshape(-1, 1)
    rectified_points1 = rectified_points1[..., :2].astype(np.int32).reshape(-1, 2)
    rectified_points2 = rectified_points2[..., :2].astype(np.int32).reshape(-1, 2)

    rectified_img_annotated = np.copy(rectified_img)

    for i, pts in enumerate(zip(rectified_points1, rectified_points2)):
        rectified_img_annotated = draw_line(
            rectified_img_annotated,
            pts,
            colors[int(i / 2)]
        )

    # Plotting and saving images
    images = [img, img_annotated, rectified_img, rectified_img_annotated]
    titles = [
        'Original',
        'Original - Annotated',
        'Rectified',
        'Rectified - Annotated'
    ]
    plot_images(
        images,
        titles,
        (2, 2),
        os.path.join(out_path, get_file_name(file))
    )

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for file in img_files:
        logging.info(f'Processing {file}:')
        direct_rectify_to_similarity(
            file,
            colors,
            args.out_path,
            args.load_annotations
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q4.yaml'
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q4'
    )
    parser.add_argument(
        '-l',
        '--load-annotations',
        help="Defines whether annotations should be loaded from file.",
        action='store_true'
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_path, 'log.txt'),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    main(args)