import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from utils.transformation_utils import (
    get_intrinsic_matrix_from_vanishing_points,
    get_vanishing_points,
)

from utils.utils import (
    get_file_name,
    get_image_annotations,
    read_image,
)

def main(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # Read config
    img_file = config['IMG']
    colors = config['COLORS']

    out_path = os.path.join(
        args.out_path, 'annotations_' + get_file_name(img_file, ext='.npy')
    )
    img = read_image(img_file)

    if args.load_annotations:
        annotations, img_annotated = np.load(out_path, allow_pickle=True)
    else:
        annotations, img_annotated = get_image_annotations(
            img,
            colors,
            n_pairs=3,
            title='Annotate 3 pairs of parallel lines that are orthogonal to ' +
                'each other.'
        )

        # 3 pairs of parallel lines.
        annotations = annotations.reshape(3, 2, 2, 2)

        np.save(out_path, (annotations, img_annotated))

    # Get intrinsics
    vanishing_points = get_vanishing_points(annotations)
    K = get_intrinsic_matrix_from_vanishing_points(vanishing_points)

    logging.info(f'Intrinsics matrix for {img_file}: \n{K}\n')

    # Plot the points
    _, axes = plt.subplots(1, 3, figsize=(12, 12))
    axes[0].set_title('Original')
    axes[0].imshow(img)
    axes[1].set_title('Annotations')
    axes[1].imshow(img_annotated)
    axes[2].set_title('Vanishing points and principal point')
    axes[2].imshow(img)

    # Vanishing points
    axes[2].fill(
        vanishing_points[:, 0],
        vanishing_points[:, 1],
        fill=False,
        edgecolor='b'
    )
    axes[2].scatter(
        vanishing_points[:, 0],
        vanishing_points[:, 1],
        c=np.array(colors)/255.
    )
    # Principal point
    axes[2].scatter(K[0, -1], K[1, -1], c='c')
    # axes[2].scatter(vanishing_points[:, 0].mean(),vanishing_points[:, 1].mean(), c='c')

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
        default='configs/q2a.yaml'
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