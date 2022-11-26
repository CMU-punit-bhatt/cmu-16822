import torch
import imageio
import numpy as np
import pytorch3d
import torch

from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
from tqdm.auto import tqdm

def generate_gif(images, path, fps=15):
    imageio.mimsave(path, images, fps=fps)


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_points_renderer(
    image_size=512,
    device=None,
    radius=0.01,
    background_color=(1, 1, 1)
):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def get_point_cloud_gif_360(
    verts,
    rgb,
    output_path='mygif.gif',
    distance=2.0,
    fov=60,
    image_size=512,
    background_color=[0., 0., 0.],
    steps=range(360, 0, -10)
):
    device = get_device()

    verts = torch.Tensor(verts)
    verts[:, 1] = - verts[:, 1]
    rgb = torch.Tensor(rgb)

    if len(verts.shape) < 3:
        verts = verts.unsqueeze(0)

    if len(rgb.shape) < 3:
        rgb = rgb.unsqueeze(0)

    renderer = get_points_renderer(
        image_size=image_size,
        background_color=background_color
    )

    point_cloud = pytorch3d.structures.Pointclouds(points=verts,
                                                   features=rgb).to(device)
    images = []

    for i in tqdm(steps):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
                                                           T=T,
                                                           fov=fov,
                                                           device=device)

        rend = renderer(point_cloud, cameras=cameras)
        images.append((rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))

    return generate_gif(images, output_path)