from pointfusion.models import GlobalFusion
from pointfusion.modalities import RGB
from pointfusion.datasets import LINEMOD
from pointfusion.loss import global_fusion
from pointfusion.trainer import Trainer

import sys
import argparse

def main(args):
    modalities = [ RGB ]
    model = GlobalFusion(modalities=modalities)
    dataset = LINEMOD(modalities=modalities)
    loss_fcn = global_fusion
    trainer = Trainer()
    trainer.batch_size = 5
    trainer.lr = 0.01
    trainer.model = model
    trainer.loss_fcn = loss_fcn
    trainer.dataset = dataset    
    trainer.fit()
    trainer.save(args.path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', default='./weights/dense_fusion_rgb.pt')
    args = parser.parse_args()
    main(args)