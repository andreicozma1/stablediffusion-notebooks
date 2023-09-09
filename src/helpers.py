"""
Various helpers for the notebooks.
Author: Andrei Cozma
"""

from enum import Enum
from io import BytesIO
from typing import List, Optional, Union
import requests
from typeguard import typechecked
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import math

import PIL
import torch
from typeguard import typechecked

mpl.rcParams["animation.embed_limit"] = 50  # MB


print(f"PyTorch version: {torch.__version__}")


class ModelsTxt2Img(str, Enum):
    SD_21 = "stabilityai/stable-diffusion-2-1"
    SD_20 = "stabilityai/stable-diffusion-2"
    SD_15 = "runwayml/stable-diffusion-v1-5"
    SD_14 = "CompVis/stable-diffusion-v1-4"


@typechecked
def plot(
    imgs: Union[PIL.Image.Image, List[PIL.Image.Image]],
    captions: Optional[Union[str, List[str]]] = None,
    n_rows: Optional[int] = None,
    fname: Optional[str] = None,
):
    """
    Plots an image or a list of images.
    If a list of images is provided, they are plotted in a grid.
    """
    if isinstance(imgs, PIL.Image.Image):
        imgs = [imgs]
    if captions is None:
        captions = [f"Image {i + 1}" for i in range(len(imgs))]
    if isinstance(captions, str):
        captions = [captions]

    batch_size = len(imgs)
    assert batch_size > 0, "No images to plot."
    assert batch_size == len(captions), "Number of images and captions must match."

    rows = n_rows or int(math.sqrt(batch_size))
    cols = int(math.ceil(batch_size / rows))
    fig = plt.figure(figsize=(cols * 4, rows * 4))

    for i, img in enumerate(imgs):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(captions[i])

        ibands, iextrema = img.getbands(), img.getextrema()
        if len(ibands) == 1:
            ax.imshow(img, cmap="gray", interpolation=None)
        else:
            ax.imshow(img, interpolation=None)

        ax.axis("off")

    fig.tight_layout()

    if fname:
        plt.savefig(f"{fname}.png" if not fname.endswith(".png") else fname)

    plt.show()
    plt.close()


@typechecked
def plot_anim(
    frames: List[Union[PIL.Image.Image, List[PIL.Image.Image]]],
    frame_captions: Optional[Union[str, List[str]]] = None,
    n_rows: Optional[int] = None,
    interval: int = 500,
    dpi: int = 75,
    embed_scale: float = 1.0,
    fname: Optional[str] = None,
):
    """
    Plots an animation from a list of frames containing images.
    Each index of `frame_imgs` and `frame_captions` corresponds to a frame/timestep.
    """
    num_frames = len(frames)
    assert num_frames > 0, "No frames to plot."
    print(f"Number of frames: {num_frames}")

    if isinstance(frames[0], PIL.Image.Image):
        frames = [[frame] for frame in frames]

    num_images_per_frame = len(frames[0])
    print(f"Number of images per frame: {num_images_per_frame}")

    if frame_captions is None:
        frame_captions = [f"Step {i + 1}" for i in range(num_frames)]

    if isinstance(frame_captions, str):
        frame_captions = [frame_captions]

    assert num_frames == len(
        frame_captions
    ), "Number of frames and frame captions must match."

    rows = n_rows or int(math.sqrt(num_images_per_frame))
    cols = int(math.ceil(num_images_per_frame / rows))

    # fig, axs = plt.subplots(
    #     rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True, dpi=50
    # )
    fig = plt.figure(figsize=(cols * 4, rows * 4), constrained_layout=True, dpi=dpi)
    axs = []
    for i in range(num_images_per_frame):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.axis("off")
        axs.append(ax)

    def update(step):
        fig.suptitle(frame_captions[step], fontsize=12)

        for i, ax in enumerate(axs):
            frame = frames[step][i]
            ibands, iextrema = frame.getbands(), frame.getextrema()
            if len(ibands) == 1:
                ax.imshow(frame, cmap="gray", interpolation=None)
            else:
                ax.imshow(frame, interpolation=None)
            ax.set_title(f"Image {i + 1} ({str(iextrema)})")

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    anim_html = anim.to_jshtml(default_mode="once", embed_frames=True)
    embed_size_em = 50 * embed_scale
    anim_html = anim_html.replace("<img", f'<img style="height: {embed_size_em}em;"')

    if fname:
        with open(f"{fname}.html" if not fname.endswith(".html") else fname, "w") as f:
            f.write(anim_html)

    display(HTML(anim_html))
    plt.close()


@typechecked
def __rescale(img: PIL.Image.Image, rescale_factor: float):
    w, h = img.width, img.height
    nw, nh = int(rescale_factor * w), int(rescale_factor * h)
    img = img.resize((nw, nh), PIL.Image.LANCZOS)
    return img


@typechecked
def rescale(imgs: Union[PIL.Image.Image, List[PIL.Image.Image]], rescale_factor: float):
    single_img = False
    if not isinstance(imgs, list):
        imgs, single_img = [imgs], True

    for i in range(len(imgs)):
        imgs[i] = __rescale(imgs[i], rescale_factor)

    return imgs[0] if single_img else imgs


@typechecked
def __resize(img: PIL.Image.Image, size: int):
    w, h = img.width, img.height
    # preserving aspect ratio
    nw, nh = (size, int(size * h / w)) if w > h else (int(size * w / h), size)
    img = img.resize((nw, nh), PIL.Image.LANCZOS)
    return img


@typechecked
def resize(imgs: Union[PIL.Image.Image, List[PIL.Image.Image]], size: int):
    single_img = False
    if not isinstance(imgs, list):
        imgs, single_img = [imgs], True

    for i in range(len(imgs)):
        imgs[i] = __resize(imgs[i], size)

    return imgs[0] if single_img else imgs


@typechecked
def __load(url: str, size: int = 512):
    response = requests.get(url)
    init_image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    init_image = __resize(init_image, size)
    return init_image


@typechecked
def load(urls: Union[str, List[str]], size: int = 512):
    single_url = False
    if not isinstance(urls, list):
        urls, single_url = [urls], True

    imgs = [__load(url, size) for url in urls]

    return imgs[0] if single_url else imgs
