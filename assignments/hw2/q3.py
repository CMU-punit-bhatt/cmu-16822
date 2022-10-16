import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import yaml

from utils.transformation_utils import (
    get_intrinsic_matrix_from_vanishing_points,
    get_plane_equation,
    get_vanishing_points,
    get_vector_normal_to_plane,
)

from utils.utils import (
    get_file_name,
    get_image_annotations,
    get_point_annotations,
    read_image,
)


def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # Read config
    img_file = config['IMG']
    colors = config['COLORS']
    n_planes = config['N_PLANES']
    depths = config['DEPTHS']

    out_path = os.path.join(
        args.out_path, 'annotations_' + get_file_name(img_file, ext='.npy')
    )
    img = read_image(img_file)

    # Get all required annotations

    if args.load_annotations:
        parallel_line_annotations, parallel_lines_img_annotated, \
        plane_annotations, known_points = np.load(
            out_path,
            allow_pickle=True
        )
    else:
        # 3 pairs of parallel lines.
        parallel_line_annotations, \
        parallel_lines_img_annotated = get_image_annotations(
            img,
            colors[:3],
            n_pairs=3,
            title='Annotate 3 pairs of parallel lines that are orthogonal to ' +
                'each other.'
        )

        parallel_line_annotations = parallel_line_annotations.reshape(3, 2, 2, 2)

        # Annotate planes 5 points at a time - 4 corners of plane and 1 "known"
        # point with assumed depth.
        plane_annotations = []
        known_points = []

        for _ in range(n_planes):
            annotations, _ = get_point_annotations(
                img,
                colors[:5],
                n_points=5
            )

            annotations = np.hstack((annotations, np.ones((5, 1))))

            assert annotations.shape == (5, 3)

            plane_annotations.append(annotations[: -1])
            known_points.append(annotations[-1])

        plane_annotations = np.array(plane_annotations)
        known_points = np.array(known_points)

        np.save(
            out_path,
            (
                parallel_line_annotations,
                parallel_lines_img_annotated,
                plane_annotations,
                known_points
            )
        )

    plane_annotations = np.load('data/q3/q3.npy', allow_pickle=True)
    plane_annotations = np.concatenate((plane_annotations, np.ones((5, 4, 1))), axis=-1)

    known_points, _ = get_point_annotations(
        img,
        colors[:5],
        n_points=5
    )
    known_points = np.concatenate((known_points, np.ones((5, 1))), axis=-1)

    # Get K
    assert parallel_line_annotations.shape == (3, 2, 2, 2)

    vanishing_points = get_vanishing_points(parallel_line_annotations)
    K = get_intrinsic_matrix_from_vanishing_points(vanishing_points)
    K_inv = np.linalg.inv(K)

    logging.info(f'Intrinsics matrix for {img_file}: \n{K}\n')

    plane_normals = []

    for annotations in plane_annotations:
        plane_normals.append(get_vector_normal_to_plane(annotations, K))


    # Figure out which pixexl lies in what plane.
    planes_annotated_img = np.copy(img)
    plane_per_pixel_img = np.zeros_like(img, dtype=np.uint8)
    plane_per_pixel = np.ones_like(img[..., 0]) * -1

    for i, (annotations, color) in enumerate(zip(plane_annotations, colors)):
        cv2.fillPoly(
            plane_per_pixel_img,
            np.array([annotations[..., :2]], dtype=np.int32),
            color=color
        )
        cv2.fillPoly(
            plane_per_pixel,
            np.array([annotations[..., :2]], dtype=np.int32),
            color=i
        )
        cv2.polylines(
            planes_annotated_img,
            np.array([annotations[..., :2]], dtype=np.int32),
            isClosed=True,
            color=color,
            thickness=5,
        )

    plt.imshow(plane_per_pixel_img)
    plt.savefig(
        os.path.join(args.out_path, 'planes_' + get_file_name(img_file)),
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

    plt.imshow(planes_annotated_img)
    plt.savefig(
        os.path.join(args.out_path, 'planes_annotated_' + get_file_name(img_file)),
        bbox_inches='tight'
    )
    plt.close()

    pixels_x, pixels_y = np.meshgrid(
        np.arange(img.shape[1]),
        np.arange(img.shape[0])
    )
    pixels = np.stack((pixels_x, pixels_y, np.ones_like(pixels_x)), axis=-1)

    real_points_3d = np.ones_like(img) * -1

    for i, (annotations, known_point, depth) in enumerate(
        zip(plane_annotations, known_points, depths)
    ):
        if i == 5:
            break

        plane = get_plane_equation(annotations, K, known_point, depth)
        valid_pixels = pixels[plane_per_pixel == i].reshape(-1, 3, 1)

        points_3d = (K_inv @ valid_pixels).reshape(-1, 3)

        lambdas = - plane[-1] / (
            plane[0] * points_3d[..., 0] + plane[1] * points_3d[..., 1] + \
            plane[2] * points_3d[..., 2]
        )
        lambdas = lambdas.reshape(-1, 1)

        real_points_3d[plane_per_pixel == i] = lambdas * points_3d

    filtered_real_points_3d = real_points_3d[
        np.all(real_points_3d != -1, axis=-1)
    ].reshape(-1, 3)

    img_colors = img[np.all(real_points_3d != -1, axis=-1)].reshape(-1, 3) / 255.

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(
    #     filtered_real_points_3d[:, 0],
    #     filtered_real_points_3d[:, 1],
    #     filtered_real_points_3d[:, 2],
    #     c=img_colors)
    # plt.show()

    marker_data = go.Scatter3d(
        x=filtered_real_points_3d[:,0],
        y=filtered_real_points_3d[:,1],
        z=filtered_real_points_3d[:,2],
        marker=dict(
            size=2,
            color=img_colors,
            opacity=0.8
        ),
        mode='markers'
    )
    fig=go.Figure(data=marker_data)
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path',
        help="Defines the config file path.",
        type=str,
        default='configs/q3.yaml'
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q3'
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