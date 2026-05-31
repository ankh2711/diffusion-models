import argparse
import os
import torch
import numpy as np
from PIL import Image

from guided_diffusion import O_DDIMSampler
from guided_diffusion.script_util import (
    model_defaults,
    diffusion_defaults,
    create_model,
    create_gaussian_diffusion,
    select_args,
)

from utils import normalize_image, save_image
from utils.config import Config
from utils.logger import logging_info
from guided_diffusion import dist_util


def prepare_model(conf, device):
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)

    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=O_DDIMSampler,
    )

    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path),
            map_location="cpu"
        ),
        strict=False,
    )

    unet.to(device)
    unet.eval()

    return unet, sampler


def run_copaint(image_path, mask_path, model_path, outdir):

    config = Config(default_config_file="CoPaint2/configs/celebahq.yaml", use_argparse=False)

    config.model_path = model_path
    config.input_image = image_path
    config.mask = mask_path
    config.outdir = outdir
    config.n_samples = 1
    config.n_iter = 1
    config.algorithm = "o_ddim"

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    unet, sampler = prepare_model(config, device)

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("1")

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)

    mask = np.array(mask).astype(np.float32)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

    image = image.to(device)
    mask = mask.to(device)

    model_kwargs = {
        "gt": image,
        "gt_keep_mask": mask
    }

    shape = (1, 3, config.image_size, config.image_size)

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return unet(x, t, None, gt=gt)

    result = sampler.p_sample_loop(
        model_fn,
        shape=shape,
        model_kwargs=model_kwargs,
        device=device,
        progress=True,
        return_all=True,
        conf=config
    )

    sample = normalize_image(result["sample"])

    os.makedirs(outdir, exist_ok=True)

    output_path = os.path.join(outdir, "copaint_restored.png")
    save_image(sample[0], output_path)

    print("Saved:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--outdir", default="result/copaint")

    args = parser.parse_args()

    run_copaint(
        args.input,
        args.mask,
        args.model_path,
        args.outdir
    )