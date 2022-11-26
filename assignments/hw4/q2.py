# Read 2d point correspondences for 1st 2 cameras
# Read P matrices
# Triangulate and get 3d points
# For each new camera pair except 1
#   read 2d-2d cam with camera  for which 3d point is available
#   calculate P_i
#   use P and P_i and triangulate all correspondences

import argparse
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os
import time
import yaml

from utils.transformation_utils import (
    get_E_8_points_alg,
    get_KRT_from_P,
    get_P_from_KRT,
    get_P_from_incremental_sfm,
    get_projection_matrices_from_E,
    triangulate
)
from utils.utils import (
    get_file_name,
    read_file,
    read_image,
)
from utils.viz_utils import get_point_cloud_gif_360

def main(args):

    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # Loading configs
    config_name = config['NAME']
    corresp_configs = config['CORRESPONDENCES']

    # Getting values
    imgs = {}
    projection_matrices = {}

    # Function to load from dict.
    def load(dict, key, path, func=None):
        if key not in dict.keys():
            if path is None:
                return None
            dict[key] = read_file(path)

            if func is not None:
                dict[key] = func(dict[key])

        return dict[key]

    points_2d_3d = {}
    points_3d = []
    colors_3d = []

    for corresp in corresp_configs:
        key1 = corresp['KEY1']
        key2 = corresp['KEY2']
        points1 = read_file(corresp['POINTS1'])
        points2 = read_file(corresp['POINTS2'])

        P1 = load(projection_matrices, key1, corresp['CAM1'], get_P_from_KRT)
        P2 = load(projection_matrices, key2, corresp['CAM2'], get_P_from_KRT)

        assert P1 is not None or P2 is not None

        if P1 is None:
            P1 = get_P_from_incremental_sfm(
                points2,
                points1,
                (
                    np.concatenate(points_2d_3d[key2]['POINTS_2D']),
                    np.concatenate(points_2d_3d[key2]['POINTS_3D'])
                )
            )

            projection_matrices[key1] = P1
        elif P2 is None:
            P2 = get_P_from_incremental_sfm(
                points1,
                points2,
                (
                    np.concatenate(points_2d_3d[key1]['POINTS_2D']),
                    np.concatenate(points_2d_3d[key1]['POINTS_3D'])
                )
            )
            projection_matrices[key2] = P2

        img1 = load(imgs, key1, corresp['IMG1'])
        img2 = load(imgs, key2, corresp['IMG2'])

        logging.info(f'Projection matrix for {config_name}_{key1}: \n{P1}\n')
        logging.info(
            f'Camera matrices for {config_name}_{key1}: ' +
            f'\n{get_KRT_from_P(P1)}\n'
        )
        logging.info(f'Projection matrix for {config_name}_{key2}: \n{P2}\n')
        logging.info(
            f'Camera matrices for {config_name}_{key2}: ' +
            f'\n{get_KRT_from_P(P2)}\n'
        )

        points_3d_i = triangulate(points1, points2, P1, P2)

        if key1 not in points_2d_3d.keys():
            points_2d_3d[key1] = {'POINTS_2D': [], 'POINTS_3D': []}
        if key2 not in points_2d_3d.keys():
            points_2d_3d[key2] = {'POINTS_2D': [], 'POINTS_3D': []}

        points_2d_3d[key1]['POINTS_2D'].append(points1)
        points_2d_3d[key1]['POINTS_3D'].append(points_3d_i)
        points_2d_3d[key2]['POINTS_2D'].append(points2)
        points_2d_3d[key2]['POINTS_3D'].append(points_3d_i)

        points_3d.append(points_3d_i)
        colors_3d.append(img1[points1[:, 1], points1[:, 0]])

        points_3d_plt = np.concatenate(points_3d)
        colors_3d_plt = np.concatenate(colors_3d) / 255.

        print(f'Creating gif for {key1}-{key2}:')
        get_point_cloud_gif_360(
            points_3d_plt[..., :3],
            colors_3d_plt,
            os.path.join(
                args.out_path,
                get_file_name(
                    '_'.join([config_name, key1, key2]),
                    '.gif'
                )
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q2.yaml'
    )
    parser.add_argument(
        '-o',
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q2'
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