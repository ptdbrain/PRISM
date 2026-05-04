# Kế hoạch triển khai: Giai đoạn 3 (Custom RTN Kernel) & Giai đoạn 4 (Benchmark)

Sau khi thành công tích hợp AutoGPTQ kernel ở Giai đoạn 2, chúng ta sẽ bước vào Giai đoạn 3 (Tối ưu hóa) và Giai đoạn 4 (Đo lường).

## Giai đoạn 3: Viết Kernel fused chuyên biệt cho RTN

Thuật toán lượng tử hóa của PRISM là **Symmetric RTN** (không có zero-point). 
Tuy nhiên, `AutoGPTQ` kernel hiện tại được thiết kế cho GPTQ (Asymmetric), do đó nó phải nạp vào bộ nhớ và xử lý một tensor vô nghĩa là `qzeros` (ta đang giả lập bằng `scale * qmax`).

### Giải pháp Custom RTN Kernel
Chúng ta sẽ rẽ nhánh (fork) AutoGPTQ kernel thành một **RTN-specific CUDA Kernel** với những tối ưu độc quyền:
1. **Loại bỏ hoàn toàn tham số `zeros`**: Tiết kiệm VRAM bandwidth, bỏ bớt 1 phép đọc global memory mỗi chu kỳ tính toán.
2. **Shift giá trị tĩnh vào LUT (Look-up Table)**:
   - *AutoGPTQ*: Nạp byte unsigned vào `deq2` LUT, sau đó tính động `W = LUT_val * scale - zero`.
   - *RTN Kernel*: Nạp thẳng giá trị *signed* (đã trừ `qmax` lúc khởi tạo) vào LUT. 
   - Công thức dequant rút gọn thành: `W = LUT_val * scale` (chỉ 1 phép nhân).
3. **Giảm thiểu Instruction**: Chuyển từ `__hfma2(LUT_val, scale, zero)` thành phép nhân `__hmul2(LUT_val, scale)`, giúp giảm instruction count và chu kỳ GPU clock.

### Triển khai
- **[NEW] `prism/kernels/rtn/rtn_kernel.cu`**: Custom kernel C++.
- **[NEW] `prism/kernels/rtn/build.py`**: JIT Compiler cho RTN kernel.
- **[NEW] `prism/runtime/rtn_kernel.py`**: Linear wrapper `RTNCustomLinear` cho 2/3/4-bit.
- Cập nhật `prism/runtime/backends.py` và `assemble.py` để bổ sung priority: **Marlin > RTN Custom > AutoGPTQ > GEMM**.

---

## Giai đoạn 4: Benchmark Framework

Dựa trên kịch bản benchmark của `AMQ/amq_speed_benchmark.py`, chúng ta sẽ xây dựng module đánh giá hiệu năng chính thức cho PRISM.

### Metrics mới & So sánh
1. **Search Time Comparison**: 
   - PRISM: Đo thời gian chạy end-to-end `profiling` + `assignment` (dự kiến tính bằng giây/phút).
   - So sánh trực quan sự đột phá về mặt tốc độ search so với AMQ (vài giờ).
2. **Tokens per Second (TPS) / GEMV / GEMM**:
   - Sử dụng thư viện `time` và GPU sync event để đo.
   - Test TPS khi batch_size=1 (Inference/Generation) và GEMM khi batch_size > 1 (Prefill).
3. **Memory Footprint**:
   - Báo cáo % VRAM tiết kiệm được sau lượng tử hóa hỗn hợp (mixed-precision).

### Triển khai
- **[NEW] `prism/benchmark/speed.py`**: Chứa logic benchmark TPS, Memory.
- **[NEW] `prism/benchmark/e2e.py`**: Chạy toàn bộ luồng từ Profiling -> Assembly -> Inference Benchmark để xuất báo cáo JSON/Terminal tương tự AMQ.

> [!IMPORTANT]
> **User Review Required:**
> 1. Cách tối ưu LUT và lược bỏ `qzeros` trong CUDA kernel là cực kỳ lợi hại, bạn có đồng ý phương án thiết kế kernel C++ như trên không?
> 2. Về phần benchmark, bạn muốn báo cáo (report) hiển thị trực tiếp trên Terminal hay lưu ra file `benchmark.json` như AMQ? Tôi dự định sẽ in đẹp (pretty-print) trên Terminal và kết xuất file JSON song song.
