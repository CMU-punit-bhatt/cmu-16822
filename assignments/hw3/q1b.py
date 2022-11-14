import argparse
import logging
import matplotlib.pyplot as plt
import os
import yaml

from utils.transformation_utils import (
    get_F_7_points_alg,
    get_F_8_points_alg,
    get_epipolar_error,
    get_epipolar_line,
    get_essential_matrix,
    homogenize
)
from utils.utils import (
    draw_correspondences,
    draw_line_eq,
    get_file_name,
    get_point_annotations,
    read_file,
    read_image
)

def main(args):

    # Getting parameter info and required data from config and other files.
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    config_name = config['NAME']
    img1_file, img2_file = config['IMGS']
    colors = config['COLORS']
    n_points = config['N_ANNOTATION_POINTS']
    correspondences = read_file(config['CORRESPONDENCES'])
    intrinsics = read_file(config['INTRINSICS'])

    assert n_points == len(colors)

    img1, img2 = read_image(img1_file), read_image(img2_file)
    points1, points2 = correspondences['pts1'], correspondences['pts2']
    K1, K2 = intrinsics['K1'], intrinsics['K2']

    img_corresp_annotated = draw_correspondences(img1, img2, points1, points2)

    plt.imshow(img_corresp_annotated)
    plt.title('Correspondences Visualization')
    plt.savefig(
        os.path.join(args.out_path, 'corresp_' + get_file_name(config_name)),
        bbox_inches='tight'
    )
    plt.close()

    # Getting fundamental matrices.
    F = get_F_7_points_alg(points1, points2)

    logging.info(f'Fundamental matrix for {config_name}: \n{F}\n')

    annotated_points, img1_annotated = get_point_annotations(
        img1,
        colors,
        n_points=n_points
    )
    epipolar_lines = get_epipolar_line(
        homogenize(annotated_points.reshape(-1, 2)),
        F
    )

    img2_annotated = draw_line_eq(img2, epipolar_lines, colors)

    # Logging essential matrix
    E = get_essential_matrix(F, K1, K2)
    logging.info(f'Essential matrix for {config_name}: \n{E}\n')

    # Plot the points
    _, axes = plt.subplots(1, 2, figsize=(12, 12))
    axes[0].set_title('View1 (Points)')
    axes[0].imshow(img1_annotated)
    axes[1].set_title('View2 (Epipolar Lines)')
    axes[1].imshow(img2_annotated)

    plt.savefig(
        os.path.join(args.out_path, get_file_name(config_name)),
        bbox_inches='tight'
    )
    plt.show()

    return F, E


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q1b_toybus.yaml'
    )
    parser.add_argument(
        '-o',
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q1b'
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