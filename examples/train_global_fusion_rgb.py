import sys

sys.path.append('../')

import pointfusion

def main():
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

if __name__ == '__main__':
    main()