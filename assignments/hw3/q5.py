import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import (
    get_F_7_points_alg,
    get_F_8_points_alg,
    get_epipolar_error,
    get_epipolar_line,
    homogenize,
    ransac_F,
)
from utils.utils import (
    compute_correspondences,
    draw_correspondences,
    draw_line_eq,
    draw_point,
    draw_points,
    get_file_name,
    read_image
)

def main(args):

    # Getting parameter info and required data from config and other files.
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    config_name = config['NAME']
    alg_type = config['ALG_TYPE']
    img1_file, img2_file = config['IMGS']
    thresh = config['THRESH']

    img1, img2 = read_image(img1_file), read_image(img2_file)

    points1, points2 = compute_correspondences(img1, img2)

    assert points1.shape == points2.shape
    assert points1.shape[-1] == 2

    img_corresp_annotated = draw_correspondences(img1, img2, points1, points2)

    plt.imshow(img_corresp_annotated)
    plt.title('Correspondences Visualization')
    plt.savefig(
        os.path.join(args.out_path, 'corresp_' + get_file_name(config_name)),
        bbox_inches='tight'
    )
    plt.close()

    n_corrs = 8 if alg_type == 'eight' else 7

    # for n_iter in n_iters:
    F, inliers, n_inliers_list = ransac_F(
        get_F_8_points_alg if alg_type == 'eight' else get_F_7_points_alg,
        get_epipolar_error,
        points1,
        points2,
        1000,
        thresh,
        n_corrs,
    )

    best_F = get_F_8_points_alg(
        points1[inliers],
        points2[inliers]
    )

    logging.info(f'Fundamental matrix for {config_name}: \n{best_F}\n')

    rand_n = np.random.randint(0, inliers.shape[0], n_corrs)

    img1_annotated = np.copy(img1)
    img1_annotated = draw_points(
        img1_annotated,
        points1[inliers[rand_n]],
        [[0, 255, 0] for _ in range(n_corrs)]
    )
    epipolar_lines = get_epipolar_line(
        homogenize(points1[inliers[rand_n]]),
        F
    )
    img2_annotated = draw_line_eq(
        img2,
        epipolar_lines,
        [[0, 255, 0] for _ in epipolar_lines]
    )

    _, axes = plt.subplots(1, 2, figsize=(12, 12))
    plt.title(config_name)
    axes[0].set_title('View1 (Points)')
    axes[0].imshow(img1_annotated)
    axes[1].set_title('View2 (Epipolar Lines)')
    axes[1].imshow(img2_annotated)

    plt.savefig(
        os.path.join(args.out_path, get_file_name(config_name)),
        bbox_inches='tight'
    )
    # plt.show()
    plt.close()

    plt.plot(n_inliers_list[:, 0], n_inliers_list[:, 1] * 100 / points1.shape[0], '-')
    plt.title(config_name)
    plt.xlabel('#Iterations')
    plt.ylabel("%\Inliers")
    plt.savefig(
        os.path.join(args.out_path, 'iters_' + get_file_name(config_name)),
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q5.yaml'
    )
    parser.add_argument(
        '-o',
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q5'
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