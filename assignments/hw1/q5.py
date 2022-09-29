import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import get_composite_img
from utils.utils import (
    get_file_name,
    get_point_annotations,
    plot_images,
    read_image
)

def overlay_multiple_images(files, colors, out_path, load_annotations=False):
    persp_img = read_image(files[0])
    final_img = np.copy(persp_img)

    for pfile in files[1:]:
        orig_img = read_image(pfile)

        if load_annotations:
            orig_points, persp_img_annotated, persp_points = np.load(
                os.path.join(
                    out_path, 'annotations_' + get_file_name(pfile, ext='.npy')
                ),
                allow_pickle=True
            )
        else:
            orig_points, _ = get_point_annotations(
                orig_img,
                colors,
                n_points=4
            )
            orig_points =  np.hstack(
                (orig_points, np.ones((4, 1)))
            ).astype(np.int32)

            persp_points, persp_img_annotated = get_point_annotations(
                final_img,
                colors,
                n_points=4
            )
            persp_points =  np.hstack(
                (persp_points, np.ones((4, 1)))
            ).astype(np.int32)

            np.save(
                os.path.join(
                    out_path, 'annotations_' + get_file_name(pfile, ext='.npy')
                ),
                (orig_points, persp_img_annotated, persp_points)
            )

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

        final_img = get_composite_img(orig_img, H, final_img)

        # Plotting and saving images
        images = [orig_img, persp_img_annotated, final_img]
        titles = [
            'Normal',
            'Perspective - Annotated Corners',
            'Warped and Overlaid Image',
        ]
        plot_images(
            images,
            titles,
            (3, 1),
            os.path.join(out_path, get_file_name(pfile))
        )

    # Plotting and saving images
    images = [persp_img, final_img]
    titles = [
        'Normal',
        'Warped and Overlaid Image',
    ]
    plot_images(
        images,
        titles,
        (2, 1),
        os.path.join(out_path, get_file_name(files[0]))
    )

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for files in img_files:
        logging.info(f'Processing {files[0]}:')
        overlay_multiple_images(
            files,
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
        default='configs/q5.yaml'
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q5'
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