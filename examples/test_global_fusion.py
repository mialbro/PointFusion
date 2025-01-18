from pointfusion import D455, Inference
from pointfusion.enums import FusionMethod
from pointfusion.models import GlobalFusion, DenseFusion

import os
import sys
import argparse
from pathlib import Path
import torch

def main(args):
    inference = Inference()
    inference.camera = D455()
    inference.model = GlobalFusion()
    #inference.model.load_state_dict(torch.load(args.weights, inference.device, weights_only=True))
    inference.model.to(inference.device)
    inference.model.eval()
    for rgb_image, depth_image, _ in inference.camera:
        scores, corners = inference(rgb_image, depth_image)

if __name__ == '__main__':
    path = os.path.join(Path(__file__).resolve().parent.parent, 'weights', 'global_fusion.pt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=path)
    args = parser.parse_args()
    main(args)