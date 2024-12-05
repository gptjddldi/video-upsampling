import argparse
import time
import cv2
from upsample import Upsample
from upsample_gpu import UpsampleGPU

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='cat')
parser.add_argument("--target_size", type=int, nargs=2, default=(2048, 2048))
args = parser.parse_args()

#padding by replication을 수행
def replication_padding(source):
    target = np.zeros((source.shape[0] + 2, source.shape[1] + 2, 3))
    target[1: source.shape[0] + 1, 1: source.shape[1] + 1, :] = source
    
    target[0] = target[1] #위 
    target[target.shape[0]-1] = target[target.shape[0]-2] #아래
    target[:, 0] = target[:,1] #왼쪽
    target[:, target.shape[1]-1] = target[:, target.shape[1]-2] #오른쪽
    
    return target

def upsample_bilinear(source, target_size):
    # -------------------------------------------------------------------------
    # Implement the upsaming algorithm with bilienar interpolation
    # Args:
    #     source (numpy array): (old_h, old_w, 3)
    #     target_size (tuple): (2,) 
    #     -- target_size[0]: new_h 
    #     -- target_size[1]: new_w
    # Return:
    #     target (numpy.ndarray): (new_h, new_w, 3)
    # -------------------------------------------------------------------------
    
    old_h, old_w, _ = source.shape #기존 이미지 크기
    new_h, new_w = target_size #새로운 이미지 크기
    
    #이전 좌표를 대응시키기 위한 계수
    a_h = old_h/new_h
    b_h = (a_h - 1)/2
    
    a_w = old_w/new_w
    b_w = (a_w - 1)/2
    
    target = np.zeros((new_h, new_w, 3))
    
    source = replication_padding(source)#source를 패딩 해 주어야함
    
    for x_new in range(new_h):
        for y_new in range(new_w):
            #대응되는 이전 좌표를 계산
            x_old = a_h*x_new + b_h
            y_old = a_w*y_new + b_w
            
            #==============================================================
            #주위 픽셀들의 값을 이용하여 새로운 픽셀의 값을 계산하는 알고리즘
            #==============================================================
            
            #참고할 점들의 index를 계산
            idx = [(int(x_old), int(y_old)), (int(x_old+1), int(y_old)), (int(x_old), int(y_old+1)), (int(x_old+1), int(y_old+1))]
            
            #가중합에 사용할 weight를 계산(합은 자동으로 1이됨)
            weight = [(int(x_old+1)-x_old)*(int(y_old+1)-y_old), (x_old-int(x_old))*(int(y_old+1)-y_old),
                      (int(x_old+1)-x_old)*(y_old-int(y_old)), (x_old-int(x_old))*(y_old-int(y_old))]
            
            #가중합을 구하여 저장
            target[x_new, y_new, :] = np.sum([w*source[i[0],i[1],:] for i,w in zip(idx, weight)], axis = 0)
    
    return target


def main(args):
    # Load image
    source_path = f'{args.image_name}.png'
    source = cv2.imread(source_path)

    # x8 upsampling with bilinear interpolation
    target_bilinear = upsample_bilinear(source, args.target_size)
    target_bilinear_path = f'{args.image_name}_bilinear.png'
    cv2.imwrite(target_bilinear_path, target_bilinear)

if __name__ == "__main__":
    main(args)
