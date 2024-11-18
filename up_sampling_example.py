import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='cat')
parser.add_argument("--target_size", type=int, nargs=2, default=(2048, 2048))
args = parser.parse_args()

def upsample_bilinear(source_image, dest_size):
    # ----------------------------------------------------------------------------------------------------------
    # Implement the upsaming algorithm with bilienar interpolation
    # Args:
    #     source_image (numpy array): (old_h, old_w, 3)
    #     dest_size (tuple): (2,) 
    #     -- dest_size[0]: new_h 
    #     -- dest_size[1]: new_w
    # Return:
    #     dest_image (numpy.ndarray): (new_h, new_w, 3)
    # ----------------------------------------------------------------------------------------------------------
    
    src_h, src_w = source_image.shape[:2] #기존 이미지 크기
    dest_h, dest_w = dest_size #새로운 이미지 크기
    
    # ----------------------------------------------------------------------------------------------------------
    # dest좌표에따라 대응되는 source 이미지에서의 좌표를 계산 하기위한 두개의 일차함수를 만든다.
    # slope_w * dest_w + intercept_w = src_w
    # slope_h * dest_w + intercept_h = src_h
    # 형태로 계산 할 것이다.
    # 기울기와 절편은 
    # (-0.5, -0.5) -> (-0.5, -0.5), (dest_w-0.5, dest_h-0.5) -> (src_w-0.5, src-0.5) 임을 이용하여 찾는다.
    # ----------------------------------------------------------------------------------------------------------
    
    slope_h = src_h/dest_h
    intercept_h = (slope_h - 1)/2
    
    slope_w = src_w/dest_w
    intercept_w = (slope_w - 1)/2
    
    dest_image = np.zeros((dest_h, dest_w, 3)) #변환된 이미지 생성

    source_image = np.pad(source_image, pad_width=((1, 1), (1, 1), (0, 0)), mode='edge') #source_image를 1칸 padding해줌(replication padding)
    
    # ----------------------------------------------------------------------------------------------------------
    # 1. dest_image각 픽셀을 돌며 dest_image에 대응되는 source_image의 좌표를 계산한다.
    # 2. 해당 좌표에 맞는 픽셀값을 bilinear interpolation을 이용하여 계산한다.
    # ----------------------------------------------------------------------------------------------------------
    for y_dest in range(dest_h):
        for x_dest in range(dest_w):
            # 1.대응되는 source_image의 좌표를 계산
            x_src = slope_w * x_dest + intercept_w
            y_src = slope_h * y_dest + intercept_h
            
            # 2.bilinear interpolation
            x0, y0 = int(x_src), int(y_src)
            x1, y1 = x0 + 1, y0 + 1
            
            # 가중치 계산을 위한 길이
            wx = x_src - x0
            wy = y_src - y0
            
            # 네 개의 이웃 픽셀 값 가져오기
            v1 = source_image[y0, x0]
            v2 = source_image[y0, x1]
            v3 = source_image[y1, x0]
            v4 = source_image[y1, x1]

            dest_image[y_dest, x_dest] = (1 - wx) * (1 - wy) * v1 + wx * (1 - wy) * v2 + (1 - wx) * wy * v3 + wx * wy * v4
    
    return dest_image


def main(args):
    # Load image
    source_path = f'./data/{args.image_name}.png'
    source_image = cv2.imread(source_path)

    # x8 upsampling with bilinear interpolation
    target_bilinear = upsample_bilinear(source_image, args.target_size)
    target_bilinear_path = f'./data/{args.image_name}_bilinear.png'
    cv2.imwrite(target_bilinear_path, target_bilinear)

if __name__ == "__main__":
    main(args)
