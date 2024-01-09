import sys

sys.path.append('../')

import pointfusion

def main():
    model_name = pointfusion.ModelName.GlobalFusion
    modalities = [ pointfusion.Modality.POINT_CLOUD ]

    model = pointfusion.GlobalFusion(point_count=400, modalities=modalities)
    dataset = pointfusion.LINEMOD(point_count=400, model_name=model_name)
    loss_fcn = pointfusion.loss.global_fusion

    trainer = pointfusion.Trainer()
    trainer.batch_size = 5
    trainer.lr = 0.01
    trainer.weight_decay = 0.001
    trainer.model = model
    trainer.loss_fcn = loss_fcn
    trainer.dataset = dataset
    
    trainer.fit()

if __name__ == '__main__':
    main()