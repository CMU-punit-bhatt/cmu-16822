import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import (
    get_homography_matrix,
    get_intrinsic_matrix_from_homographies,
    get_vector_normal_to_plane,
)

from utils.utils import (
    get_file_name,
    get_point_annotations,
    read_image,
)

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # Read config
    img_file = config['IMG']
    colors = config['COLORS']
    orig_points = np.asanyarray(config['ORIG_POINTS'])
    n_homographies = int(config['N_HOMOGRAPHIES'])

    out_path = os.path.join(
        args.out_path, 'annotations_' + get_file_name(img_file, ext='.npy')
    )
    img = read_image(img_file)
    orig_points =  np.concatenate(
        (orig_points, np.ones((*orig_points.shape[: 2   ], 1))),
        axis=-1
    ).astype(np.int32)

    if args.load_annotations:
        annotations, imgs_annotated = np.load(out_path, allow_pickle=True)
    else:
        annotations = []
        imgs_annotated = []
        for i in range(n_homographies):
            persp_points, img_annotated = get_point_annotations(
                img,
                colors,
                n_points=4
            )

            persp_points =  np.hstack(
                (persp_points, np.ones((4, 1)))
            ).astype(np.int32)

            annotations.append(persp_points)
            imgs_annotated.append(img_annotated)

        np.save(out_path, (annotations, imgs_annotated))

    assert len(annotations) == n_homographies == len(imgs_annotated)

    # Get intrinsic matrix
    homographies = [
        get_homography_matrix(orig_points[i], persp_points)
        for i, persp_points in enumerate(annotations)
    ]
    K = get_intrinsic_matrix_from_homographies(homographies)

    logging.info(f'Intrinsics matrix for {img_file}: \n{K}\n')

    # Log angle between planes.
    # Get direction of normal for each
    normals = [
        get_vector_normal_to_plane(annotations[i], K)
        for i in range(n_homographies)
    ]

    for i in range(n_homographies):
        for j in range(i + 1, n_homographies):
            angle = np.rad2deg(np.arccos(np.dot(normals[i], normals[j])))
            logging.info(
                f'Angle between Plane {i + 1} and Plane {j + 1}: ' +
                f'{angle} or {180 - angle}')


    for annotations, img_annotated, color in zip(
        annotations,
        imgs_annotated,
        colors
    ):
        cv2.fillPoly(
            img_annotated,
            np.array([annotations[..., :2]], dtype=np.int32),
            color=(255, 255, 255)
        )


    # Plot the points
    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].set_title('Original')
    axes[0, 0].imshow(img)
    axes[0, 1].set_title('Annotated Square 1')
    axes[0, 1].imshow(imgs_annotated[0])
    axes[1, 0].set_title('Annotated Square 2')
    axes[1, 0].imshow(imgs_annotated[1])
    axes[1, 1].set_title('Annotated Square 3')
    axes[1, 1].imshow(imgs_annotated[2])

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
        default='configs/q2b.yaml'
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/q2'
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