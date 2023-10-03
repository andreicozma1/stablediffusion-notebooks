"""
Various helpers for the notebooks.
Author: Andrei Cozma
"""

import math
import os
from enum import Enum
from io import BytesIO
from typing import List, Optional, Union

import diffusers
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
import torch
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from PIL import Image
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
    imgs: Union[Image.Image, List[Image.Image]],
    captions: Optional[Union[str, List[str]]] = None,
    n_rows: Optional[int] = None,
    fname: Optional[str] = None,
):
    """
    Plots an image or a list of images.
    If a list of images is provided, they are plotted in a grid.
    """
    if isinstance(imgs, Image.Image):
        imgs = [imgs]

    n_imgs = len(imgs)
    assert n_imgs > 0, "No images to plot."

    if captions is None:
        captions = [f"Image {i + 1}" for i in range(len(imgs))]
    if isinstance(captions, str):
        captions = [captions]

    assert n_imgs == len(
        captions
    ), f"Number of images and captions must match. Got {n_imgs} images and {len(captions)} captions."

    rows = n_rows or int(math.sqrt(n_imgs))
    cols = int(math.ceil(n_imgs / rows))
    fig = plt.figure(figsize=(cols * 4, rows * 4))

    for i, img in enumerate(imgs):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(captions[i])

        ibands, iextrema = img.getbands(), img.getextrema()
        if len(ibands) == 1:
            ax.imshow(img, cmap="gray", interpolation="none")
        else:
            ax.imshow(img, interpolation="none")

        ax.axis("off")

    fig.tight_layout()

    if fname:
        plt.savefig(fname if fname.endswith(".png") else f"{fname}.png")

    plt.show()
    plt.close()


@typechecked
def plot_anim(
    frames: List[Union[Image.Image, List[Image.Image]]],
    frame_titles: Optional[Union[str, List[str]]] = None,
    captions: Optional[Union[str, List[str]]] = None,
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
    n_frames = len(frames)
    assert n_frames > 0, "No frames to plot."
    print(f"Number of frames: {n_frames}")

    if isinstance(frames[0], Image.Image):
        frames = [[frame] for frame in frames]

    if frame_titles is None:
        frame_titles = [f"Step {i + 1}" for i in range(n_frames)]

    if isinstance(frame_titles, str):
        frame_titles = [frame_titles]

    assert n_frames == len(
        frame_titles
    ), f"Number of frames and frame titles must match. Got {n_frames} frames and {len(frame_titles)} frame titles."

    if captions is None:
        captions = [f"Image {i + 1}" for i in range(len(frames[0]))]
    if isinstance(captions, str):
        captions = [captions]

    n_imgs_per_frame = len(frames[0])
    print(f"Number of images per frame: {n_imgs_per_frame}")

    assert n_imgs_per_frame == len(
        captions
    ), f"Number of images per frame and captions must match. Got {n_imgs_per_frame} images per frame and {len(captions)} captions."

    rows = n_rows or int(math.sqrt(n_imgs_per_frame))
    cols = int(math.ceil(n_imgs_per_frame / rows))

    fig = plt.figure(figsize=(cols * 4, rows * 4), constrained_layout=True, dpi=dpi)
    axs = []
    for i in range(n_imgs_per_frame):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.axis("off")
        axs.append(ax)

    def update(step):
        fig.suptitle(frame_titles[step], fontsize=12)
        for i, ax in enumerate(axs):
            img = frames[step][i]
            ax.set_title(captions[i])

            ibands, iextrema = img.getbands(), img.getextrema()
            if len(ibands) == 1:
                ax.imshow(img, cmap="gray", interpolation="none")
            else:
                ax.imshow(img, interpolation="none")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval)
    anim_html = anim.to_jshtml(default_mode="once", embed_frames=True)
    embed_size_em = 50 * embed_scale
    anim_html = anim_html.replace("<img", f'<img style="height: {embed_size_em}em;"')

    if fname:
        # Save as HTML
        with open(f"{fname}.html" if not fname.endswith(".html") else fname, "w") as f:
            f.write(anim_html)
        # Save as GIF
        anim.save(
            f"{fname}.gif" if not fname.endswith(".gif") else fname,
            writer="imagemagick",
        )

    display(HTML(anim_html))
    plt.close()


@typechecked
def __rescale_image(img: Image.Image, rescale_factor: float):
    w, h = img.width, img.height
    nw, nh = int(rescale_factor * w), int(rescale_factor * h)
    img = img.resize((nw, nh), Image.LANCZOS)
    return img


@typechecked
def rescale_image(imgs: Union[Image.Image, List[Image.Image]], rescale_factor: float):
    single_img = False
    if not isinstance(imgs, list):
        imgs, single_img = [imgs], True

    for i in range(len(imgs)):
        imgs[i] = __rescale_image(imgs[i], rescale_factor)

    return imgs[0] if single_img else imgs


@typechecked
def __resize_image(img: Image.Image, size: int):
    w, h = img.width, img.height
    # preserving aspect ratio
    nw, nh = (size, int(size * h / w)) if w > h else (int(size * w / h), size)
    img = img.resize((nw, nh), Image.LANCZOS)
    return img


@typechecked
def resize_image(imgs: Union[Image.Image, List[Image.Image]], size: int = 512):
    single_img = False
    if not isinstance(imgs, list):
        imgs, single_img = [imgs], True

    for i in range(len(imgs)):
        imgs[i] = __resize_image(imgs[i], size)

    return imgs[0] if single_img else imgs


@typechecked
def load_image(urls: Union[str, List[str]], size: int = 512):
    single_url = False
    if not isinstance(urls, list):
        urls, single_url = [urls], True

    imgs = [diffusers.utils.load_image(url) for url in urls]
    imgs = [resize_image(img, size) for img in imgs]

    return imgs[0] if single_url else imgs


def save_tmp_outputs(
    imgs: List[Image.Image],
    basedir: str = "output_imgs",
    subdir: Optional[str] = None,
):
    """
    Saves a list of images to a temporary directory.
    The directory is cleared every time the function is called.
    """
    savepath = os.path.join(basedir, subdir) if subdir else basedir
    os.makedirs(savepath, exist_ok=True)
    img_fname_template = "img_{:04d}.png"
    # clear any old existing images
    for fname in os.listdir(savepath):
        if fname.startswith("img_") and fname.endswith(".png"):
            fpath = os.path.join(savepath, fname)
            os.remove(fpath)
    # save the new images
    for i, img in enumerate(imgs):
        fpath = os.path.join(savepath, img_fname_template.format(i))
        img.save(fpath)
