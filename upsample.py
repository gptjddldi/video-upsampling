import math
from multiprocessing import Pool, cpu_count
import numpy as np

class Upsample:
    def __init__(self, target_size, num_processes):
        self.target_size = target_size
        self.num_processes = min([num_processes, cpu_count()])

    @staticmethod
    def replication_padding(source):
        return np.pad(source, ((1, 1), (1, 1), (0, 0)), mode='edge')
    
    @staticmethod
    def process(args):
        """각 프로세스가 처리할 연산을 수행하는 정적 메서드"""
        source, start_x, end_x, old_h, old_w, new_h, new_w = args
        
        # 계수 계산
        a_h = old_h/new_h
        b_h = (a_h - 1)/2
        a_w = old_w/new_w
        b_w = (a_w - 1)/2
        
        # 결과를 저장할 배열 생성
        chunk_height = end_x - start_x
        target_chunk = np.zeros((chunk_height, new_w, 3))
        
        # 패딩된 소스 이미지
        padded_source = Upsample.replication_padding(source)
        
        # 할당된 x 범위에 대해 처리
        for x_idx, x_new in enumerate(range(start_x, end_x)):
            for y_new in range(new_w):
                # 대응되는 이전 좌표를 계산
                x_old = a_h*x_new + b_h
                y_old = a_w*y_new + b_w
                
                # 참고할 점들의 index를 계산
                idx = [
                    (int(x_old), int(y_old)),
                    (int(x_old+1), int(y_old)),
                    (int(x_old), int(y_old+1)),
                    (int(x_old+1), int(y_old+1))
                ]
                
                # 가중합에 사용할 weight를 계산
                weight = [
                    (int(x_old+1)-x_old)*(int(y_old+1)-y_old),
                    (x_old-int(x_old))*(int(y_old+1)-y_old),
                    (int(x_old+1)-x_old)*(y_old-int(y_old)),
                    (x_old-int(x_old))*(y_old-int(y_old))
                ]
                
                # 가중합을 구하여 저장
                target_chunk[x_idx, y_new, :] = np.sum(
                    [w*padded_source[i[0],i[1],:] for i,w in zip(idx, weight)],
                    axis=0
                )
        
        return target_chunk
    
    def process_upsample(self, source):
        """입력 이미지를 주어진 target_size로 upsampling"""
        if not isinstance(source, np.ndarray):
            raise TypeError("입력은 NumPy 배열이어야 합니다.")
        
        old_h, old_w, _ = source.shape
        new_h, new_w = self.target_size
        
        # 각 프로세스가 처리할 x축 범위 계산
        chunk_size = math.ceil(new_h / self.num_processes)
        chunks = [
            (i * chunk_size, min((i + 1) * chunk_size, new_h))
            for i in range(self.num_processes)
        ]
        
        # 각 프로세스에 전달할 인자 생성
        process_args = [
            (source, start_x, end_x, old_h, old_w, new_h, new_w)
            for start_x, end_x in chunks
        ]
        
        # 멀티프로세싱으로 처리
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process, process_args)
        
        # 결과 조합
        return np.concatenate(results, axis=0)
