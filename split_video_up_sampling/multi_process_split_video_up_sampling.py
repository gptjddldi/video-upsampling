import argparse
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Process

parser = argparse.ArgumentParser()
parser.add_argument('--video_name', type=str)
parser.add_argument('--partial_count', type=int, default=4)
parser.add_argument('--scale_factor', type=int, default=1)
args = parser.parse_args()

#이미지의 크기를 바꾼다.
def resize_frame(frame, scale_factor=4, interpolation=cv2.INTER_LANCZOS4):
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 프레임 크기 조정 (리사이징)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    return resized_frame

def enhance_frame_quality_0(frame, scale_factor_1=4):

    # Step 1: Resize the image using bicubic interpolation
    resized = resize_frame(frame, scale_factor=scale_factor_1, interpolation=cv2.INTER_CUBIC)

    # Step 2: Apply a sharpening filter to enhance edges
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_frame = cv2.filter2D(resized, -1, sharpening_kernel)

    # Step 3: (Optional) Reduce noise with a bilateral filter
    enhanced_frame = cv2.bilateralFilter(sharpened_frame, d=9, sigmaColor=75, sigmaSpace=75)

    return enhanced_frame

def enhance_frame_quality_1(frame, scale_factor=4):
    resized = resize_frame(frame, scale_factor)
    
    # 선명도 개선
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # 대비 향상
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame

def upscale_frame(frame, scale_factor=2):
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_frame

def partial_video_upscale(video_name, start_frame, end_frame, segment_index, new_width, new_height):
    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print("캡처 실패")
        exit()

    # 동영상 라이터 초기화
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(f'output_segment_{segment_index}.mp4', fourcc, fps, (new_width, new_height))

    # 분할한 프레임 만큼만 이미지 업스케일링 적용
    print(f"{segment_index}번째 세그먼트 시작")
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        enhanced_frame = enhance_frame_quality_0(frame, args.scale_factor)

        out.write(enhanced_frame)

    print(f'{segment_index}번째 세그먼트 생성 완료')

    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main(args):
    start_full_time = time.time()

    start_time = time.time()

    # 동영상 캡처 초기화
    cap = cv2.VideoCapture(args.video_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    segment_frames = total_frames // args.partial_count
    list_of_procs = []

    for r in range(args.partial_count):
        start_frame = r * segment_frames
        # 마지막 분할은 마지막 프레임 까지 읽도록 조건을 추가
        end_frame = (r + 1) * segment_frames if r < args.partial_count - 1 else total_frames
        p = Process(target=partial_video_upscale, args=(args.video_name, start_frame, end_frame, r + 0, 
                                              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale_factor),
                                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale_factor)))
        list_of_procs.append(p)
        p.start()

    for p in list_of_procs:
        p.join()

    end_time = time.time()
    print(f"동영상 분활 후 처리까지의 실행 시간: {end_time - start_time:.5f} 초")

    # 동영상을 다시 하나로 합치기 위한 코드 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale_factor) 
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale_factor)
    out = cv2.VideoWriter(f'output.mp4', fourcc, fps, (new_width, new_height))
    # 분할한 영상을 다시 하나로 합침
    start_time = time.time()
    for r in range(args.partial_count):
        cap = cv2.VideoCapture(f'output_segment_{r}.mp4') 
        if not cap.isOpened(): 
            print(f"첫 번째 동영상을 열 수 없습니다: {f'output_segment_{r}.mp4'}") 
            return
        
        while True: 
            ret, frame = cap.read() 
            if not ret: 
                break 
            out.write(frame)

    end_time = time.time()
    end_full_time = time.time()

    print(f"동영상 합치는데 걸리는 시간: {end_time - start_time:.5f} 초")
    print(f"사용한 총 시간: {end_full_time - start_full_time:.5f} 초")

    cap.release()

if __name__ == "__main__":
    main(args)
