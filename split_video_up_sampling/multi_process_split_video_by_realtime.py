import argparse
import cv2
import numpy as np
import time
import multiprocessing as mp

import multi_process_split_video_up_sampling as my_split_video
import upsample
import upsample_gpu

def resize_frame(frame, scale_factor=4, interpolation=cv2.INTER_LANCZOS4):
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 프레임 크기 조정 (리사이징)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    return resized_frame

def enhance_frame_quality_0(index, frame, scale, result_dict):
    resized = resize_frame(frame, scale_factor=scale, interpolation=cv2.INTER_CUBIC)
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_frame = cv2.filter2D(resized, -1, sharpening_kernel)
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
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale_factor)
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale_factor)
    out = cv2.VideoWriter(f'output.mp4', fourcc, fps, (new_width, new_height))

    lock = mp.Lock()
    manager = mp.Manager()
    result_dict = manager.dict()
    
    print("시작")
    start_time = time.time()
    temp = False

    pool = mp.Pool(args.process_count)

    while not temp:
        tasks = []
        
        frame_count = 0
        for i in range(args.process_count):
            ret, frame = read_frame(cap, lock)
            if not ret:
                temp = True
                break
            tasks.append(pool.apply_async(enhance_frame_quality_0, (frame_count + i, frame, args.scale_factor, result_dict)))

        for task in tasks:
            task.get()

        for i in range(frame_count, frame_count + len(tasks)):
            out.write(result_dict[i])


    end_time = time.time()
    print("완료")
    print(f"사용한 총 시간: {end_time - start_time:.5f} 초")

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str)
    parser.add_argument('--process_count', type=int, default=4)
    parser.add_argument('--scale_factor', type=int, default=2)
    args = parser.parse_args()
    main(args)
