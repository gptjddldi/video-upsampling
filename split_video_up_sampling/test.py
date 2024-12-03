import cv2
import multiprocessing as mp
import time

def read_frames(input_video, frame_queue, frame_queue_lock):
    cap = cv2.VideoCapture(input_video)
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_queue_lock:
            frame_queue.put((frame_index, frame))
        frame_index += 1
    cap.release()
    with frame_queue_lock:
        frame_queue.put(None)  # End signal for all processes

def upscale_and_write(frame_queue, output_queue, frame_queue_lock, output_queue_lock, scale):
    while True:
        with frame_queue_lock:
            frame_data = frame_queue.get()
        if frame_data is None:  # End signal
            with output_queue_lock:
                output_queue.put(None)
            break
        frame_index, frame = frame_data
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        upscaled_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        with output_queue_lock:
            output_queue.put((frame_index, upscaled_frame))

def write_frames(output_queue, output_video, total_processes, output_queue_lock):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 첫 번째 프레임을 가져와서 VideoWriter 설정하기
    while True:
        with output_queue_lock:
            frame_data = output_queue.get()
        if frame_data is not None:
            frame_index, frame = frame_data
            break

    frame_height, frame_width = frame.shape[:2]
    out = cv2.VideoWriter(output_video, fourcc, 30, (frame_width, frame_height))

    frames_buffer = {}
    end_signals_received = 0

    if frame_data is not None:
        frames_buffer[frame_index] = frame

    while True:
        with output_queue_lock:
            frame_data = output_queue.get()
        if frame_data is None:
            end_signals_received += 1
            if end_signals_received == total_processes:
                break
        else:
            frame_index, frame = frame_data
            frames_buffer[frame_index] = frame

    for i in sorted(frames_buffer.keys()):
        out.write(frames_buffer[i])

    out.release()

def main():
    input_video = 'input.mp4'
    output_video = 'output.mp4'
    scale = 2
    total_processes = 2

    frame_queue = mp.Queue()
    output_queue = mp.Queue()
    frame_queue_lock = mp.Lock()
    output_queue_lock = mp.Lock()

    processes = []

    print("시작")
    start_time = time.time()

    for i in range(total_processes):
        p = mp.Process(target=read_frames, args=(input_video, frame_queue, frame_queue_lock))
        processes.append(p)        
        p.start()

    for p in processes:
        p.join()
    
    processes = []

    for i in range(total_processes):
        p = mp.Process(target=write_frames, args=(output_queue, output_video, total_processes, output_queue_lock))
        processes.append(p)        
        p.start()
        
    for p in processes:
        p.join()

    end_time = time.time()

    print(f"사용한 총 시간: {end_time - start_time:.5f} 초")
    print('Video upscaled and saved to', output_video)

if __name__ == "__main__":
    main()
