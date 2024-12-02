import argparse
import time
import cv2
from upsample import Upsample
from upsample_gpu import UpsampleGPU

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='cat')
parser.add_argument("--target_size", type=int, nargs=2, default=(2048, 2048))
args = parser.parse_args()


def main(args):
    source_path = f'{args.image_name}.png'
    source = cv2.imread(source_path)
    
    for i in [8]:
        start = time.time()
        upsampler = Upsample(args.target_size, i)
        ret = upsampler.process_upsample(source)
        cv2.imwrite(f'{args.image_name}_bilinear{i}.png', ret)
        print(f'num_processes: {i}, elapsed time: {time.time() - start:.4f}')
    
    gpu_sampler = UpsampleGPU(args.target_size)
    start = time.time()
    ret = gpu_sampler.process_upsample(source)
    cv2.imwrite(f'{args.image_name}_bilinear_gpu.png', ret)
    print(f'GPU, elapsed time: {time.time() - start:.4f}')

if __name__ == "__main__":
    main(args)
