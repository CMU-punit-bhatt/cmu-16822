import argparse
import logging
import os

from tqdm.auto import tqdm
from utils.utils import read_image
from utils.viz_utils import generate_gif

name = 'bottle'
in_path = os.path.join('data/q3/bottle/gif')
name_fmt = 'frame%06d.png'
total = 400
out_path = 'output/q3/'

os.makedirs(out_path, exist_ok=True)

imgs = []

for i in tqdm(range(total)):
    imgs.append(read_image(os.path.join(in_path, name_fmt%i)))

generate_gif(imgs, os.path.join(out_path, name + '.gif'), fps=100)
