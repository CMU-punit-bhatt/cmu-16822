import argparse
import logging
import os
import yaml

from q1 import rectify_to_affinity
from q2 import rectify_to_similarity
from q3 import overlay_image
from q4 import direct_rectify_to_similarity
from q5 import overlay_multiple_images

def q1(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for file in img_files:
        logging.info(f'Processing {file}:')
        rectify_to_affinity(
            file,
            colors,
            args.out_path,
            args.load_annotations
        )

def q2(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for file in img_files:
        logging.info(f'Processing {file}:')
        rectify_to_similarity(
            file,
            colors,
            args.out_path,
            args.load_annotations
        )

def q3(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for file_pair in img_files:
        logging.info(f'Processing {file_pair[0]}-{file_pair[1]}:')
        overlay_image(
            file_pair,
            colors,
            args.out_path,
            args.load_annotations
        )

def q4(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for file in img_files:
        logging.info(f'Processing {file}:')
        direct_rectify_to_similarity(
            file,
            colors,
            args.out_path,
            args.load_annotations
        )

def q5(args):
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    img_files = config['SRC']
    colors = config['COLORS']

    for files in img_files:
        logging.info(f'Processing {files[0]}:')
        overlay_multiple_images(
            files,
            colors,
            args.out_path,
            args.load_annotations
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-q',
        '--question',
        help="Defines the question number.",
        required=True,
        type=int
    )
    parser.add_argument(
        '-c',
        '--config-path',
        help="Defines the config file path.",
        type=str,
        required=True
    )
    parser.add_argument(
        '--out-path',
        help="Defines the output file path.",
        type=str,
        default='output/'
    )
    parser.add_argument(
        '-l',
        '--load-annotations',
        help="Defines whether annotations should be loaded from file.",
        action='store_true'
    )
    args = parser.parse_args()

    # Adding question to the output path
    args.out_path = os.path.join(args.out_path, str(args.question))

    os.makedirs(args.out_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_path, 'log.txt'),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    if args.question == 1:
        q1(args)
    elif args.question == 2:
        q2(args)
    elif args.question == 3:
        q3(args)
    elif args.question == 4:
        q4(args)
    elif args.question == 5:
        q5(args)
    else:
        print('Incorrect question number!')

