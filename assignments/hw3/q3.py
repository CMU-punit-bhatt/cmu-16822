import argparse
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import yaml

from utils.transformation_utils import (
    triangulate
)
from utils.utils import (
    draw_correspondences,
    get_file_name,
    read_file,
    read_image
)

def main(args):

    # Getting parameter info and required data from config and other files.
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    config_name = config['NAME']
    img1_file, img2_file = config['IMGS']
    correspondences = config['CORRESPONDENCES']
    projection_matrices = config['PROJECTIONS']

    # assert n_points == len(colors)

    img1, img2 = read_image(img1_file), read_image(img2_file)
    points1 = read_file(correspondences[0])
    points2 = read_file(correspondences[1])
    P1 = read_file(projection_matrices[0])
    P2 = read_file(projection_matrices[1])

    assert points1.shape == points2.shape

    img_corresp_annotated = draw_correspondences(img1, img2, points1, points2)

    plt.imshow(img_corresp_annotated)
    plt.title('Correspondences Visualization')
    plt.savefig(
        os.path.join(args.out_path, 'corresp_' + get_file_name(config_name)),
        bbox_inches='tight'
    )
    plt.close()

    points_3d = triangulate(points1, points2, P1, P2)

    img_colors = img1[points1[:, 1], points1[:, 0]].reshape(-1, 3) / 255.

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(
    #     points_3d[:, 0],
    #     points_3d[:, 1],
    #     points_3d[:, 2],
    #     c=img_colors)
    # plt.show()

    marker_data = go.Scatter3d(
        x=points_3d[:,0],
        y=points_3d[:,1],
        z=points_3d[:,2],
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
        default='configs/q3.yaml'
    )
    parser.add_argument(
        '-o',
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q3'
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