import sys
import argparse

sys.path.append('../')

import pointfusion

def main(args):
    model_name = pointfusion.ModelName.GlobalFusion
    modalities = [ pointfusion.Modality.RGB ]
    model = pointfusion.GlobalFusion(modalities=modalities)
    dataset = pointfusion.LINEMOD(model_name=model_name, modalities=modalities)
    loss_fcn = pointfusion.loss.global_fusion
    trainer = pointfusion.Trainer()
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