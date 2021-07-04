import argparse
import cv2
import numpy as np
import sys
import time

import caffe

import torch
from torchvision.transforms import transforms
import onnxruntime

from target_model import mobilenet_v1

class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

def load_pytorch_image(image_path):
    img = cv2.imread(image_path)

    transform_normalize = NormalizeGjz(mean=127.5, std=128)
    transform_to_tensor = ToTensorGjz()
    transform = transforms.Compose([
        transform_to_tensor, 
        transform_normalize
    ])
    img = transform(img).unsqueeze(0)
    return img

def load_model_pytorch(factor):
    if factor == 1:
        model = mobilenet_v1.mobilenet_1()
        checkpoint = 'target_model/mb1_120x120.pth'
    elif factor == 0.5:
        model = mobilenet_v1.mobilenet_05()
        checkpoint = 'target_model/mb05_120x120.pth'

    cp = torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()

    for k in cp.keys():
        kc = k.replace('module.', '')
        if kc in model_dict.keys():
            model_dict[kc] = cp[k]
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = cp[k]

    model.load_state_dict(model_dict)
    return model

def pytorch_forward(model, img):
    model.eval()
    t_start = time.time()
    output = model(img)
    t_end = time.time()
    torch.set_printoptions(precision=7)
    return output, t_end - t_start

def load_onnx_image(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img = (img - 127.5) / 128.
    return img

def load_model_onnx(factor):
    if factor == 1:
        session = onnxruntime.InferenceSession('target_model/mb1_120x120.onnx', None)
    elif factor == 0.5:
        session = onnxruntime.InferenceSession('target_model/mb05_120x120.onnx', None)
    return session

def onnx_forward(session, img):
    inp_dct = {'input': img}
    t_start = time.time()
    output = session.run(None, inp_dct)[0]
    t_end = time.time()
    return output, t_end - t_start

def load_caffe_image(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img = (img - 127.5) / 128.
    return img

def load_model_caffe(version):
    deploy_file = ''
    model_file = ''

    if version == 1:
        deploy_file = 'caffe_model/mobilenet1.prototxt'
        model_file = 'caffe_model/mobilenet1.caffemodel'
    elif version == 0.5:
        deploy_file = 'caffe_model/mobilenet05.prototxt'
        model_file = 'caffe_model/mobilenet05.caffemodel'
    
    caffe.set_mode_cpu()
    model = caffe.Net(deploy_file, model_file, caffe.TEST)
    return model

def caffe_forward(model, img):
    model.blobs['blob1'].data[...] = img
    t_start = time.time()
    output = model.forward()
    t_end = time.time()
    return output['fc_blob1'], t_end - t_start

def main(args):
    if args.factor != 1 and args.factor != 0.5:
        sys.exit(-1)

    pytorch_model = load_model_pytorch(args.factor)
    onnx_model = load_model_onnx(args.factor)
    caffe_model = load_model_caffe(args.factor)

    pytorch_image = load_pytorch_image(args.image)
    onnx_image = load_onnx_image(args.image)
    caffe_image = load_caffe_image(args.image)

    pytorch_output, t_pytorch = pytorch_forward(pytorch_model, pytorch_image)
    onnx_output, t_onnx = onnx_forward(onnx_model, onnx_image)
    caffe_output, t_caffe = caffe_forward(caffe_model, caffe_image)

    output = pytorch_output.detach().numpy()
    print(output[0][0:5])
    print(onnx_output[0][0:5])
    print(caffe_output[0][0:5])

    print(t_pytorch * 1000)
    print(t_onnx * 1000)
    print(t_caffe * 1000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference of pytorch model and caffe model')
    parser.add_argument('-f', '--factor', type=float, default=1)
    parser.add_argument('-i', '--image', type=str, default='test_data/front.jpg')
    args = parser.parse_args()
    main(args)