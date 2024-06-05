import sys
sys.path.append('../')

import argparse
import pointfusion

def main(args):
    camera = pointfusion.D455()
    inference = pointfusion.Inference(args.weights)

    for (color, depth, point_cloud) in camera:
        scores, corners = inference.predict(color, depth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./weights/dense_fusion_rgb.pt')
    args = parser.parse_args()
    main(args)