import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import(
    calculate_test_lines_angle,
    cosine,
    get_affine_rectification_H,
    get_rectified_lines,
    get_rectified_points,
    get_similarity_rectification_H,
    warp_image,
)
from utils.utils import (
    draw_line,
    get_file_name,
    get_image_annotations,
    plot_images,
    read_image
)

def rectify_to_similarity(file, colors, out_path, load_annotations=False):

    img = read_image(file)

    if load_annotations:
        affine_annotations, annotations, \
        affine_rectified_img_annotated = np.load(
            os.path.join(
                out_path, 'annotations_' + get_file_name(file, ext='.npy')
            ),
            allow_pickle=True
        )

        affine_rectification_H = get_affine_rectification_H(affine_annotations)
        affine_rectified_img, _ = warp_image(img, affine_rectification_H)
    else:
        # Getting affine-rectification annotations
        affine_annotations, _ = get_image_annotations(
            img,
            colors,
            n_pairs=2,
            title='Mark 2 Pairs of Parallel Lines'
        )

        # Getting annotations - (2, 2, 2, 2).
        affine_annotations = affine_annotations.reshape(2, 2, 2, 2)

        affine_rectification_H = get_affine_rectification_H(affine_annotations)
        affine_rectified_img, _ = warp_image(img, affine_rectification_H)

        annotations, \
        affine_rectified_img_annotated = get_image_annotations(
            affine_rectified_img,
            colors,
            n_pairs=2,
            title='Mark 2 Pairs of Orthogonal Lines'
        )
        annotations = annotations.reshape(2, 2, 2, 2)

        np.save(
            os.path.join(
                out_path, 'annotations_' + get_file_name(file, ext='.npy')
            ),
            (affine_annotations, annotations, affine_rectified_img_annotated)
        )

    rectification_H = get_similarity_rectification_H(annotations)

    rectified_img, rectification_Ht = warp_image(
        affine_rectified_img,
        rectification_H
    )

    # Getting homogenous points - (2, 2, 3) each
    affine_rectified_points1 = np.concatenate(
        (annotations[:, :, 0], np.ones((2, 2, 1))),
        axis=-1
    )
    affine_rectified_points2 = np.concatenate((annotations[:, :, 1],
                                               np.ones((2, 2, 1))), axis=-1)

    # Getting n pairs of perpendicular lines - (2, 2, 3)
    affine_rectified_lines = np.cross(
        affine_rectified_points1,
        affine_rectified_points2
    )
    affine_rectified_lines /= affine_rectified_lines[:, :, -1].reshape(2, 2, 1)

    calculate_test_lines_angle(
        img,
        rectification_H @ affine_rectification_H,
        colors,
        n_pairs=2,
        out_path=os.path.join(out_path, 'test_lines_' + get_file_name(file))
    )

    # Getting perpendicular annotations on original image
    original_points1 = get_rectified_points(
        affine_rectified_points1,
        np.linalg.inv(affine_rectification_H)
    ).reshape(-1, 3)
    original_points2 = get_rectified_points(
        affine_rectified_points2,
        np.linalg.inv(affine_rectification_H)
    ).reshape(-1, 3)
    original_points1 /= original_points1[..., 2].reshape(-1, 1)
    original_points2 /= original_points2[..., 2].reshape(-1, 1)

    original_points1 = original_points1[..., :2].astype(np.int32).reshape(-1, 2)
    original_points2 = original_points2[..., :2].astype(np.int32).reshape(-1, 2)

    img_annotated = np.copy(img)

    for i, pts in enumerate(zip(original_points1, original_points2)):
        img_annotated = draw_line(
            img_annotated,
            pts,
            colors[int(i / 2)]
        )

    # Getting perpendicular annotations on final similarity rectified image.
    rectified_points1 = get_rectified_points(
        affine_rectified_points1,
        rectification_Ht.dot(rectification_H)
    ).reshape(-1, 3)
    rectified_points2 = get_rectified_points(
        affine_rectified_points2,
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
    images = [
        img,
        img_annotated,
        affine_rectified_img,
        affine_rectified_img_annotated,
        rectified_img,
        rectified_img_annotated,
    ]
    titles = [
        'Original',
        'Original - Annotated',
        'Affine-Rectified',
        'Affine-Rectified - Annotated',
        'Rectified',
        'Rectified - Annotated'
    ]
    plot_images(
        images,
        titles,
        (3, 2),
        os.path.join(out_path, get_file_name(file))
    )

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for file in img_files:
        logging.info(f'Processing {file}:')
        rectify_to_similarity(file, colors, args.out_path, args.load_annotations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q1_q2.yaml'
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q2'
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