import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import(
    calculate_test_lines_angle,
    get_affine_rectification_H,
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


def rectify_to_affinity(file, colors, out_path, load_annotations=False):

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
            n_pairs=2,
            title='Mark 2 Pairs of Parallel Lines'
        )

        # Getting annotations - (2, 2, 2, 2).
        annotations = annotations.reshape(2, 2, 2, 2)

        np.save(
            os.path.join(
                out_path, 'annotations_' + get_file_name(file, ext='.npy')
            ),
            (annotations, img_annotated)
        )

    rectification_H = get_affine_rectification_H(annotations)
    rectified_img, _ = warp_image(img, rectification_H)

    calculate_test_lines_angle(
        img,
        rectification_H,
        colors,
        n_pairs=2,
        out_path=os.path.join(out_path, 'test_lines_' + get_file_name(file))
    )

    # Getting homogenous points - (2, 2, 3) each
    points1 = np.concatenate(
        (annotations[:, :, 0], np.ones((2, 2, 1))),
        axis=-1
    )
    points2 = np.concatenate(
        (annotations[:, :, 1], np.ones((2, 2, 1))),
        axis=-1
    )

    rectified_img_annotated = np.copy(rectified_img)

    rectified_points1 = get_rectified_points(
        points1,
        rectification_H
    ).reshape(-1, 3)
    rectified_points2 = get_rectified_points(
        points2,
        rectification_H
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

    # Plotting and saving images
    images = [img, img_annotated, rectified_img, rectified_img_annotated]
    titles = [
        'Original',
        'Original - Annotated',
        'Rectified',
        'Rectified - Annotated',
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
        rectify_to_affinity(file, colors, args.out_path, args.load_annotations)


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
        default='output/q1'
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

