# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from build.convert_hf_checkpoint import convert_hf_checkpoint
from config.model_config import (
    ModelConfig,
    ModelDistributionChannel,
    resolve_model_config,
)

from requests.exceptions import HTTPError


def _download_hf_snapshot(
    model_config: ModelConfig, artifact_dir: Path, hf_token: Optional[str]
):
    from huggingface_hub import snapshot_download

    # Download and store the HF model artifacts.
    print(f"Downloading {model_config.name} from HuggingFace...")
    try:
        snapshot_download(
            model_config.distribution_path,
            local_dir=artifact_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns="*safetensors*",
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError(
                "Access denied. Run huggingface-cli login to authenticate."
            )
        else:
            raise e

    
    # Convert the model to the torchchat format.
    print(f"Converting {model_config.name} to torchchat format...")
    convert_hf_checkpoint(model_dir=artifact_dir, model_name=model_config.name, remove_bin_files=True)


def _download_direct(
    model_config: ModelConfig,
    artifact_dir: Path,
):
    for url in model_config.distribution_path:
        filename = url.split("/")[-1]
        local_path = artifact_dir / filename
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, str(local_path.absolute()))


def download_and_convert(
    model: str, models_dir: Path, hf_token: Optional[str] = None
) -> None:
    model_config = resolve_model_config(model)
    model_dir = models_dir / model_config.name

    # Download into a temporary directory. We'll move to the final location once
    # the download and conversion is complete. This allows recovery in the event
    # that the download or conversion fails unexpectedly.
    temp_dir = models_dir / "downloads" / model_config.name
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if (
            model_config.distribution_channel
            == ModelDistributionChannel.HuggingFaceSnapshot
        ):
            _download_hf_snapshot(model_config, temp_dir, hf_token)
        elif model_config.distribution_channel == ModelDistributionChannel.DirectDownload:
            _download_direct(model_config, temp_dir)
        else:
            raise RuntimeError(
                f"Unknown distribution channel {model_config.distribution_channel}."
            )
        
        # Move from the temporary directory to the intended location,
        # overwriting if necessary.
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        os.rename(temp_dir, model_dir)

    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


def is_model_downloaded(model: str, models_dir: Path) -> bool:
    model_config = resolve_model_config(model)

    # Check if the model directory exists and is not empty.
    model_dir = models_dir / model_config.name
    return os.path.isdir(model_dir) and os.listdir(model_dir)


def main(args):
    download_and_convert(args.model, args.model_directory, args.hf_token)
