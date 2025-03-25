from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="export")  # only hydra config and overrides
def main(overrides):
    # get checkpoint
    api = wandb.Api()
    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    checkpoint_path = f"{project_path}/model-{overrides.runid}:{overrides.checkpoint}"
    checkpoint = Path(api.artifact(checkpoint_path).download()) / "model.ckpt"

    # get training config and merge with overrides
    run = api.run(f"{project_path}/{overrides.runid}")
    config = OmegaConf.create(deepcopy(run.config))
    for key in overrides.deletes:  # needed because omegaconf doesn't allow deleting
        config.pop(key, None)
    with open_dict(config):
        config.merge_with(overrides)

    # get state dict
    # only network params and remove network. prefix
    state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
    state_dict = {k.replace("network.", ""): v for k, v in state_dict.items() if k.startswith("network.")}

    # save both config and state dict
    save_dir = Path(overrides.save_dir) / overrides.name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)
    torch.save(state_dict, save_dir / "state_dict.pt")


if __name__ == "__main__":
    main()
