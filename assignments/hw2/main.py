import argparse
import logging
import os
import yaml

import q1a
import q1b
import q2a
import q2b
import q3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-q',
        '--question',
        help="Defines the question number.",
        required=True,
        type=str
    )
    parser.add_argument(
        '-c',
        '--config-path',
        help="Defines the config file path.",
        type=str,
        required=True
    )
    parser.add_argument(
        '-o',
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
    # args.out_path = os.path.join(args.out_path, str(args.question))

    os.makedirs(args.out_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_path, 'log.txt'),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    if args.question == 'q1a':
        q1a.main(args)
    elif args.question == 'q1b':
        q1b.main(args)
    elif args.question == 'q2a':
        q2a.main(args)
    elif args.question == 'q2b':
        q2b.main(args)
    elif args.question == 'q2c':
        q2b.main(args)
    elif args.question == 'q3':
        q3.main(args)
    else:
        print('Incorrect question number!')

