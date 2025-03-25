from argparse import ArgumentParser
from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt_dir", type=Path)
    args = parser.parse_args()

    # get config
    with initialize(config_path=str(args.ckpt_dir), version_base=None):
        config = compose(config_name="config")  # can add overrides here

    # device and precision
    device, dtype = torch.device("cuda"), torch.float32
    torch.set_float32_matmul_precision("high")

    # get network
    # no need to trace because params already init
    network = instantiate(config.network)
    network.load_state_dict(torch.load(args.ckpt_dir / "state_dict.pt", weights_only=True))
    network.eval()
    network.to(device, dtype=dtype)

    # always compile network
    # reduce-overhead is much faster than default
    compiled_network = torch.compile(network, fullgraph=True, mode="reduce-overhead")

    # TODO: test on some fake or real data
    print("Now test on some data!")
