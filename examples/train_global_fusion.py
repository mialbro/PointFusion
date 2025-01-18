from pointfusion.models import GlobalFusion
from pointfusion.datasets import LINEMOD
from pointfusion.loss import global_fusion
from pointfusion import Trainer
from pointfusion import FusionMethod

import os
import sys
import argparse
from pathlib import Path

def main(args):
    trainer = Trainer()
    trainer.model = GlobalFusion()
    trainer.dataset = LINEMOD(fusion_method=FusionMethod.GLOBAL)
    trainer.path = args.path
    loss_fcn = global_fusion
    trainer.batch_size = 5
    trainer.lr = 0.01
    trainer.loss_fcn = loss_fcn
    trainer.fit()

if __name__ == '__main__':
    default = os.path.join(Path(__file__).resolve().parent.parent, 'weights', 'global_fusion.pt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', default=default)
    args = parser.parse_args()
    main(args)