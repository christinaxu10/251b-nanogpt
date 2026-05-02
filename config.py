import torch, json

ckpt = torch.load("checkpoint.pt", map_location="cpu")

with open("config.json", "w") as f:
    json.dump(ckpt["config"], f, indent=2)