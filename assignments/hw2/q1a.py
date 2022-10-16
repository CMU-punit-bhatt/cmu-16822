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

from utils.utils import read_image, read_file, draw_line, get_file_name

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_file = config['IMG']

    # Getting the projection matrix P
    correspondences = read_file(config['CORRESPONDENCES'])

    points_2d = correspondences[:, :2]
    points_3d = correspondences[:, 2:]

    P = get_projection_matrix_from_correspondences(points_2d, points_3d)

    logging.info(f'Projection matrix for {img_file}: \n{P}\n')

    # Getting imaged surface 3d points
    surface_points_3d = homogenize(read_file(config['SURFACE_POINTS']))
    # Reshaping for batch mat mul
    surface_points_3d = surface_points_3d.reshape(-1, 4, 1)

    # Should result in (N, 3, 1)
    surface_points_2d = np.matmul(P, surface_points_3d).reshape(-1, 3)
    surface_points_2d /= surface_points_2d[..., -1].reshape(-1, 1)

    # Getting bounding box
    box_corners = read_file(config['BOUNDING_BOX'])
    points1_3d = homogenize(box_corners[:, :3]).reshape(-1, 4, 1)
    points2_3d = homogenize(box_corners[:, 3:]).reshape(-1, 4, 1)

    points1_2d = np.matmul(P, points1_3d).reshape(-1, 3)
    points2_2d = np.matmul(P, points2_3d).reshape(-1, 3)
    points1_2d /= points1_2d[..., -1].reshape(-1, 1)
    points2_2d /= points2_2d[..., -1].reshape(-1, 1)

    img = read_image(img_file)
    box_img = np.copy(img)

    for i in range(box_corners.shape[0]):
        box_img = draw_line(
            box_img,
            (np.int32(points1_2d[i, :2]), np.int32(points2_2d[i, :2])),
            (0, 0, 255),
            thickness=3
        )

    # Plot the points
    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].set_title('Original')
    axes[0, 0].imshow(img)
    axes[0, 1].set_title('Annotations')
    axes[0, 1].imshow(img)
    axes[1, 0].set_title('Surface Points')
    axes[1, 0].imshow(img)
    axes[1, 1].set_title('Bounding Box')
    axes[1, 1].imshow(box_img)
    axes[0, 1].scatter(points_2d[:, 0], points_2d[:, 1], s=1.5)
    axes[1, 0].scatter(surface_points_2d[:, 0], surface_points_2d[:, 1], s=1e-3)

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
        default='configs/q1a.yaml'
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q1'
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