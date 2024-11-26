import pyopencl as cl
import numpy as np

class UpsampleGPU:
    def __init__(self, target_size):
        self.target_size = target_size
        self._init_opencl()
        self._build_program()

    def _init_opencl(self):
        """OpenCL 컨텍스트와 커맨드 큐 초기화"""
        platform = cl.get_platforms()[0]
        self.device = platform.get_devices(device_type=cl.device_type.GPU)[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

    def _build_program(self):
        """OpenCL 커널 프로그램 빌드"""
        kernel_code = """
        __kernel void upsample(
            __global const float* source,
            __global const float* padded_source,
            __global float* target,
            const int old_h,
            const int old_w,
            const int new_h,
            const int new_w,
            const float a_h,
            const float b_h,
            const float a_w,
            const float b_w
        ) {
            int x_new = get_global_id(0);
            int y_new = get_global_id(1);
            
            if (x_new >= new_h || y_new >= new_w) return;
            
            // 원본 이미지에서의 대응 좌표 계산
            float x_old = a_h * x_new + b_h;
            float y_old = a_w * y_new + b_w;
            
            // 주변 픽셀의 정수 좌표 계산
            int x0 = (int)x_old;
            int x1 = x0 + 1;
            int y0 = (int)y_old;
            int y1 = y0 + 1;
            
            // 가중치 계산
            float wx1 = x_old - x0;
            float wx0 = 1.0f - wx1;
            float wy1 = y_old - y0;
            float wy0 = 1.0f - wy1;
            
            // 패딩된 이미지에서의 인덱스 계산 (패딩을 고려하여 +1)
            x0 += 1;
            x1 += 1;
            y0 += 1;
            y1 += 1;
            
            int padded_w = old_w + 2;
            
            // 각 채널에 대해 보간
            for (int c = 0; c < 3; c++) {
                float p00 = padded_source[(x0 * padded_w + y0) * 3 + c];
                float p01 = padded_source[(x0 * padded_w + y1) * 3 + c];
                float p10 = padded_source[(x1 * padded_w + y0) * 3 + c];
                float p11 = padded_source[(x1 * padded_w + y1) * 3 + c];
                
                // 쌍선형 보간
                float result = wx0 * wy0 * p00 +
                             wx1 * wy0 * p10 +
                             wx0 * wy1 * p01 +
                             wx1 * wy1 * p11;
                             
                target[(x_new * new_w + y_new) * 3 + c] = result;
            }
        }
        """
        self.program = cl.Program(self.context, kernel_code).build()

    def process_upsample(self, source):
        """GPU를 사용하여 이미지 업샘플링"""
        if not isinstance(source, np.ndarray):
            raise TypeError("입력은 NumPy 배열이어야 합니다.")
        
        # 원본 크기와 목표 크기
        old_h, old_w, _ = source.shape
        new_h, new_w = self.target_size
        
        # 패딩 추가
        padded_source = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        # 계수 계산
        a_h = old_h/new_h
        b_h = (a_h - 1)/2
        a_w = old_w/new_w
        b_w = (a_w - 1)/2
        
        # 입력 배열을 float32로 변환
        source = source.astype(np.float32)
        padded_source = padded_source.astype(np.float32)
        
        # 결과를 저장할 배열 생성
        target = np.zeros((new_h, new_w, 3), dtype=np.float32)
        
        # GPU 메모리 버퍼 생성
        source_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=source)
        padded_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=padded_source)
        target_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, target.nbytes)
        
        # 커널 실행
        kernel = self.program.upsample
        kernel.set_args(
            source_buf, padded_buf, target_buf,
            np.int32(old_h), np.int32(old_w),
            np.int32(new_h), np.int32(new_w),
            np.float32(a_h), np.float32(b_h),
            np.float32(a_w), np.float32(b_w)
        )
        
        # 글로벌 워크 사이즈 설정
        global_size = (new_h, new_w)
        local_size = None  # OpenCL이 자동으로 최적화
        
        # 커널 실행
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
        
        # 결과를 호스트 메모리로 복사
        cl.enqueue_copy(self.queue, target, target_buf)
        
        return target