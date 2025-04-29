import argparse

import torch


def main(args):
    sd = torch.load(args.src, map_location="cpu")["model"]
    sd = {k: v for k, v in sd.items() if "teacher" not in k}
    sd = {
        k.replace("backbone.vision_backbone", "image_encoder"): v for k, v in sd.items()
    }
    sd = {k.replace("mlp.fc1", "mlp.layers.0"): v for k, v in sd.items()}
    sd = {k.replace("mlp.fc2", "mlp.layers.1"): v for k, v in sd.items()}
    sd = {k.replace("convs", "neck.convs"): v for k, v in sd.items()}
    sd = {
        k.replace("transformer.encoder", "memory_attention"): v for k, v in sd.items()
    }
    sd = {k.replace("maskmem_backbone", "memory_encoder"): v for k, v in sd.items()}
    sd = {k.replace("maskmem_backbone", "memory_encoder"): v for k, v in sd.items()}
    sd = {k.replace("mlp.lin1", "mlp.layers.0"): v for k, v in sd.items()}
    sd = {k.replace("mlp.lin2", "mlp.layers.1"): v for k, v in sd.items()}
    torch.save({"model": sd}, args.src.replace(".pt", "_converted.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    args = parser.parse_args()

    main(args)
