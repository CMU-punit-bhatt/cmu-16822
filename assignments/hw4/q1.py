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
    get_projection_matrices_from_E,
    triangulate
)
from utils.utils import (
    read_file,
)

def main(args):

    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    config_name = config['NAME']
    correspondences = read_file(config['CORRESPONDENCES'])
    intrinsics = read_file(config['INTRINSICS'])

    points1, points2 = correspondences['pts1'], correspondences['pts2']
    K1, K2 = intrinsics.item()['K1'], intrinsics.item()['K2']

    # Logging essential matrix
    E = get_E_8_points_alg(points1, points2, K1, K2)
    logging.info(f'Essential matrix for {config_name}: \n{E}\n')

    P1, P2 = get_projection_matrices_from_E(E, K1, K2, points1, points2)

    logging.info(f'Projection matrix 1 for {config_name}: \n{P1}\n')
    logging.info(f'Projection matrix 2 for {config_name}: \n{P2}\n')

    points_3d = triangulate(points1, points2, P1, P2)
    img_colors = np.array([[255, 0, 0] for _ in range(points_3d.shape[0])]) / 255.

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(
    #     points_3d[:, 0],
    #     points_3d[:, 1],
    #     points_3d[:, 2],
    #     c=img_colors)
    # plt.show()

    marker_data = go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        marker=dict(
            size=5,
            color=img_colors,
            opacity=1
        ),
        mode='markers'
    )
    fig=go.Figure(data=marker_data)
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q1.yaml'
    )
    parser.add_argument(
        '-o',
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