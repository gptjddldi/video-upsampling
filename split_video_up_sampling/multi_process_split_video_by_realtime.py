import argparse
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

import multi_process_split_video_up_sampling as my_split_video

parser = argparse.ArgumentParser()
parser.add_argument('--video_name', type=str)
parser.add_argument('--process_count', type=int, default=4)
parser.add_argument('--scale_factor', type=int, default=2)
args = parser.parse_args()

def enhance_frame_quality_0(index, frame, scale, result_dict):
    # Step 1: Resize the image using bicubic interpolation
    resized = my_split_video.resize_frame(frame, scale_factor=scale, interpolation=cv2.INTER_CUBIC)

    # Step 2: Apply a sharpening filter to enhance edges
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened_frame = cv2.filter2D(resized, -1, sharpening_kernel)

    # Step 3: (Optional) Reduce noise with a bilateral filter
    enhanced_frame = cv2.bilateralFilter(sharpened_frame, d=9, sigmaColor=75, sigmaSpace=75)

    result_dict[index] = enhanced_frame

def read_frame(cap, lock):
    with lock:
        ret, frame = cap.read()
    return ret, frame

def main(args):
    cap = cv2.VideoCapture(args.video_name)

    if not cap.isOpened():
        print("캡처 실패")
        exit()

    # 동영상 writer 초기화
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale_factor) 
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale_factor)
    out = cv2.VideoWriter(f'output.mp4', fourcc, fps, (new_width, new_height))

    # 업스케일링 적용
    lock = mp.Lock()
    manager = mp.Manager()
    result_dict = manager.dict()

    print(f"시작")
    start_time = time.time()
    temp = False
    frame_count = 0

    while not temp:
        list_of_procs_by_frame = []
        for i in range(args.process_count):
            ret, frame = read_frame(cap, lock)
            if not ret:
                temp = True
                break
                
            proc = mp.Process(target=enhance_frame_quality_0, args=(frame_count + i, frame, args.scale_factor, result_dict))
            proc.start()
            list_of_procs_by_frame.append(proc)

        for proc in list_of_procs_by_frame:
            proc.join()

        for i in range(frame_count, frame_count + len(list_of_procs_by_frame)):
            out.write(result_dict[i])

        frame_count += len(list_of_procs_by_frame)

    end_time = time.time()
    print(f'완료')
    print(f"사용한 총 시간: {end_time - start_time:.5f} 초")

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(args)
