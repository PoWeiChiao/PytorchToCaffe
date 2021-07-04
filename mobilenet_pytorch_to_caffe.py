import argparse
import sys
import torch
from torch.autograd import Variable
import pytorch_to_caffe
from  target_model import mobilenet_v1

def load_pretrained_model(model, checkpoint):
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()

    for k in checkpoint.keys():
        kc = k.replace('module.', '')
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = checkpoint[k]

    model.load_state_dict(model_dict)
    return model

def main(args):
    name = args.name

    if args.factor == 1:
        model = mobilenet_v1.mobilenet_1()
    elif args.factor == 0.5:
        model = mobilenet_v1.mobilenet_05()
    else:
        sys.exit(-1)

    model = load_pretrained_model(model, args.checkpoint)
    model.eval()

    input = Variable(torch.rand(1, 3, 120, 120))
    pytorch_to_caffe.trans_net(model, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer mobilenet from pytorch to caffe')
    parser.add_argument('-f', '--factor', type=float, default=1)
    parser.add_argument('-c', '--checkpoint', type=str, default='target_model/mb1_120x120.pth')
    parser.add_argument('-n', '--name', type=str, default='caffe_model/mobilenet1')
    args = parser.parse_args()
    main(args)