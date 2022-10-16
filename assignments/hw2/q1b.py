import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import (
    get_projection_matrix_from_correspondences,
    homogenize
)

from utils.utils import read_image, read_file, draw_line, get_file_name, get_point_annotations

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_file = config['IMG']
    img = read_image(img_file)
    out_path = os.path.join(
        args.out_path, 'annotations_' + get_file_name(img_file, ext='.npy')
    )

    points_3d = read_file(config['CORRESPONDENCES'])
    n_points = points_3d.shape[0]

    if args.load_annotations:
        points_2d, img_annotated = np.load(out_path, allow_pickle=True)
    else:
        points_2d, img_annotated = get_point_annotations(
            img,
            [[0, 255, 0]] * n_points,
            n_points
        )

        np.save(out_path, (points_2d, img_annotated))

    # Getting the projection matrix P
    P = get_projection_matrix_from_correspondences(points_2d, points_3d)

    logging.info(f'Projection matrix for {img_file}: \n{P}\n')

    # Getting bounding box
    box_corners = read_file(config['EDGES'])
    points1_3d = homogenize(box_corners[:, :3]).reshape(-1, 4, 1)
    points2_3d = homogenize(box_corners[:, 3:]).reshape(-1, 4, 1)

    points1_2d = np.matmul(P, points1_3d).reshape(-1, 3)
    points2_2d = np.matmul(P, points2_3d).reshape(-1, 3)
    points1_2d /= points1_2d[..., -1].reshape(-1, 1)
    points2_2d /= points2_2d[..., -1].reshape(-1, 1)

    box_img = np.copy(img)

    for i in range(box_corners.shape[0]):
        box_img = draw_line(
            box_img,
            (np.int32(points1_2d[i, :2]), np.int32(points2_2d[i, :2])),
            (255, 0, 0),
            thickness=5
        )

    # Plot the points
    _, axes = plt.subplots(1, 3, figsize=(12, 12))
    axes[0].set_title('Original')
    axes[0].imshow(img)
    axes[1].set_title('Annotations')
    axes[1].imshow(img_annotated)
    axes[2].set_title('All Lines/Edges')
    axes[2].imshow(box_img)

    plt.savefig(
        os.path.join(args.out_path, get_file_name(img_file)),
        bbox_inches='tight'
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q1b.yaml'
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