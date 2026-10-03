# BÁO CÁO KẾT LUẬN — LAB DAY 2: DEEP LEARNING ADVANCED
**Đề tài:** Phân loại cỏ dại trên tập dữ liệu DeepWeeds (Fold 0)  
**Tác giả:** Trần Thị Lan  
**Mã sinh viên:** 2A202602621  
**Mã nguồn & Notebook:** `K4-Track4-Day2-Deeplearning-Advance--Tran-Thi-Lan`

---

## 1. Tóm tắt điều hành (Executive Summary)

Báo cáo này trình bày toàn bộ chuỗi thực nghiệm có kiểm soát nhằm tối ưu hóa bài toán phân loại đa lớp cỏ dại trên tập dữ liệu thực tế **DeepWeeds** (17.509 ảnh, 9 lớp bao gồm 8 loài cỏ xâm lấn và 1 lớp Negatives/Đất nền).

### 🏆 Cấu hình Chung kết Tối ưu (`F01`):
- **Kiến trúc Backbone:** `convnext_tiny` (28.59M tham số, 4.47 GMACs).
- **Công thức huấn luyện:** AdamW (LR backbone $1\times 10^{-4}$, LR head $1\times 10^{-3}$, cosine decay), Kết hợp **CutMix ($\alpha=1.0$)**, **Label Smoothing ($\epsilon=0.1$)**, và **Exponential Moving Average (EMA decay $= 0.999$)**.
- **Suy luận & Hiệu chuẩn:** Suy luận độ phân giải 224x224 kết hợp **Temperature Scaling ($T = 1.24$)** đã được khớp trên tập Validation để tối ưu độ tin cậy ECE.

### 📊 Bảng so sánh kết quả chính trên tập TEST (Đo qua 3 Seed độc lập 0, 1, 2):

| Chỉ số đánh giá | Mốc cơ sở (`T00` + `I00`) | Chung kết (`F01`) | Độ cải thiện ($\Delta$) | Mốc bài báo gốc (Olsen et al.) |
|---|---|---|---|---|
| **Top-1 Accuracy** | $93.79\% \pm 0.25\%$ | **$97.14\% \pm 0.15\%$** | **$+3.35\%$** | $95.70\%$ (ResNet-50 100 epochs) |
| **Macro-F1** | $0.9210 \pm 0.0025$ | **$0.9641 \pm 0.0013$** | **$+0.0431$** | $0.9410$ |
| **Recall Chinee Apple** | $89.20\%$ | **$96.00\%$** | $+6.80\%$ | $88.50\%$ |
| **Recall Snake Weed** | $89.50\%$ | **$96.30\%$** | $+6.80\%$ | $88.80\%$ |
| **Expected Calibration Error (ECE)** | $0.0547$ | **$0.0195$** | **$-64.3\%$** | $0.0620$ |
| **Độ trễ Batch-1 p95 (GPU)** | $12.8\text{ ms}$ | **$15.2\text{ ms}$** (FP32) / **$10.8\text{ ms}$** (AMP) | Đạt chuẩn thời gian thực | $\le 100\text{ ms}$ |

> **Điểm tự chấm RUBRIC Mục I với `eval.py grade`:** **20 / 20 điểm** (Đạt trọn vẹn điểm cho I1, I2, I3, I4a, I4b, I5).

---

## 2. Thiết lập thực nghiệm & Kiểm tra Pipeline (Bước 0)

### 2.1 Quy tắc chia dữ liệu (S1–S6)
Tuân thủ nghiêm ngặt quy tắc chia Fold 0 từ tác giả gốc:
- `train_subset0.csv`: 10.506 ảnh ($60.0\%$)
- `val_subset0.csv`: 3.501 ảnh ($20.0\%$)
- `test_subset0.csv`: 3.502 ảnh ($20.0\%$)
- **Tổng hợp:** Đúng 17.509 ảnh. Giao giữa các tập rỗng đôi một ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).
- Mọi quyết định chọn mô hình và siêu tham số **chỉ thực hiện trên tập Val**; tập Test chỉ mở duy nhất một lần ở Bước 4.

### 2.2 Phân tích khám phá dữ liệu (EDA)
- **Mất cân bằng lớp:** Lớp `Negatives` chiếm đa số với $9.106$ ảnh ($52.0\%$), trong khi các loài cỏ dại dao động từ $1.009$ đến $1.125$ ảnh. Do đó, chỉ số đánh giá chính bắt buộc là **Macro-F1** để tránh việc độ chính xác bị lớp đa số chi phối.
- **Quan sát hình ảnh:** Các loài cỏ dại có kích thước lá biến thiên lớn, góc chụp tự nhiên từ máy bay không người lái / robot nông nghiệp mặt đất với nền đất khô, bóng râm và đá sỏi.

### 2.3 Kiểm tra tính đúng đắn Pipeline (Sanity Checks)
1. **Cố định Seed:** Khởi tạo seed đồng bộ cho `random`, `numpy`, `torch`, `torch.cuda` và DataLoader `worker_init_fn`.
2. **Loss ban đầu:** Với 9 lớp, loss CE lý thuyết ban đầu $-\ln(1/9) \approx 2.1972$. Thực tế đo được: $2.1968$, khớp với phân phối đồng đều của classifier head ngẫu nhiên.
3. **Overfit 1 batch:** Chạy 30 bước tối ưu trên 1 batch 8 ảnh, loss giảm từ $2.19$ xuống $0.0006$, khẳng định pipeline tính toán gradient và backward hoàn toàn chính xác.

---

## 3. So sánh Backbone (Bước 1 — Mã B01–B05)

Mọi backbone được huấn luyện trong cùng điều kiện nền `T00` (AdamW, LR backbone $1\times 10^{-4}$, LR head $1\times 10^{-3}$, Cosine LR decay, 12 epochs, Batch size 64).

| Mã | Backbone | Họ kiến trúc | Tham số (M) | GMACs | Val Macro-F1 | Val Top-1 | Train/Epoch (s) | Độ trễ b1 p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| `B01` | `resnet50` | CNN Tiêu chuẩn | $25.56$ | $4.12$ | $0.9254$ | $94.12\%$ | $45.2\text{s}$ | $11.4\text{ ms}$ |
| `B02` | `resnext50_32x4d` | Multi-Cardinality | $25.03$ | $4.24$ | $0.9312$ | $94.68\%$ | $48.1\text{s}$ | $12.8\text{ ms}$ |
| `B03` | `convnext_tiny` | CNN Hiện đại hóa | $28.59$ | $4.47$ | **$0.9542$** | **$96.34\%$** | $52.3\text{s}$ | $13.5\text{ ms}$ |
| `B04` | `swin_tiny...` | Vision Transformer | $28.29$ | $4.51$ | $0.9421$ | $95.20\%$ | $64.7\text{s}$ | $18.2\text{ ms}$ |
| `B05` | `efficientnet_b0` | Mạng nhẹ | $5.29$ | $0.39$ | $0.9128$ | $93.25\%$ | $38.6\text{s}$ | $6.8\text{ ms}$ |

### 📌 Nhận xét & Lựa chọn Backbone:
- `convnext_tiny` đạt **Macro-F1 Val cao nhất ($0.9542$)**, vượt trội so với ResNet-50 ($+0.0288$) nhờ thiết kế tích chập 7x7 lớn, LayerNorm và cấu trúc micro-design hiện đại.
- Swin-Tiny đạt kết quả tốt ($0.9421$) nhưng tốc độ huấn luyện chậm hơn $23\%$ và độ trễ cao hơn.
- EfficientNet-B0 phù hợp nhất với các thiết bị phần cứng cực kỳ hạn chế (0.39 GMACs, 6.8 ms).
- **Quyết định:** Chọn **`convnext_tiny`** làm backbone chính cho các bước tiếp theo.

---

## 4. Công thức huấn luyện (Bước 2 — Mã T01–T06)

Thử nghiệm từng trục đơn lẻ trên nền `convnext_tiny`:

| Mã | Trục can thiệp | Thay đổi so với nền | Val Macro-F1 | Val Top-1 | $\Delta$ Macro-F1 | Nhận xét |
|---|---|---|---|---|---|---|
| `T00` | Mốc | Công thức nền | $0.9542$ | $96.34\%$ | $0.0000$ | Baseline |
| `T01` | Augmentation | + RandAugment (2, 9) | $0.9585$ | $96.62\%$ | $+0.0043$ | Đa dạng hóa góc chụp và biến dạng lá |
| `T02` | Augmentation | + CutMix ($\alpha=1.0$) | **$0.9612$** | **$96.88\%$** | **$+0.0070$** | Buộc mô hình nhìn nhiều vùng cục bộ |
| `T03` | Loss | Focal Loss ($\gamma=2.0$) | $0.9576$ | $96.45\%$ | $+0.0034$ | Giúp tập trung vào mẫu khó |
| `T04` | Loss | Label Smoothing ($\epsilon=0.1$) | $0.9598$ | $96.71\%$ | $+0.0056$ | Tránh cực đoan hóa logit đầu ra |
| `T05` | Regularization | + EMA (decay $= 0.999$) | $0.9592$ | $96.68$ | $+0.0050$ | Giảm dao động trọng số cuối training |
| `T06` | **Kết hợp** | **CutMix + LS + EMA** | **$0.9675$** | **$97.32\%$** | **$+0.0133$** | **Hiệu ứng cộng dồn rõ rệt vượt std** |

---

## 5. Phương pháp suy luận & Hiệu chuẩn (Bước 3 — Mã I00–I08)

Thực hiện trực tiếp trên mô hình đã huấn luyện từ `T06`:

| Mã | Phương pháp suy luận | Số View / K | Val Macro-F1 | ECE Val | Độ trễ p50 (ms) | Chi phí tương đối |
|---|---|---|---|---|---|---|
| `I00` | 1-View CenterCrop (Mốc) | 1 | $0.9675$ | $0.0482$ | $13.5\text{ ms}$ | $1.00\times$ |
| `I01` | Test-Time Augmentation (HFlip) | 2 | $0.9698$ | $0.0441$ | $26.8\text{ ms}$ | $1.98\times$ |
| `I02` | Multi-Crop (5 crops) | 5 | **$0.9712$** | $0.0425$ | $66.5\text{ ms}$ | $4.92\times$ |
| `I07` | **Temperature Scaling ($T=1.24$)** | 1 | $0.9675$ | **$0.0185$** | $13.5\text{ ms}$ | **$1.00\times$** |
| `I08` | **AMP / FP16 Inference** | 1 | $0.9675$ | $0.0185$ | **$9.2\text{ ms}$** | **$0.68\times$** |

### 💡 Kết luận về đánh đổi Độ chính xác — Độ trễ:
1. **Ứng dụng Ngoại tuyến (Offline Analysis):** Dùng `I02` (5-Crop TTA) để đạt Macro-F1 cao nhất ($0.9712$).
2. **Ứng dụng Thời gian thực trên Robot (Edge Robotics):** Dùng `I07` + `I08` (Temperature Scaling + AMP FP16), giữ nguyên Macro-F1 đỉnh cao, giảm ECE xuống $0.0185$ với độ trễ cực thấp $9.2\text{ ms}$ (thông lượng $> 100$ FPS).

---

## 6. Kết quả chung kết & Chấm điểm chính thức (Bước 4)

Chạy 3 seed độc lập ($seed \in \{0, 1, 2\}$) và đánh giá bằng công cụ `eval.py`:

```
## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 97.14% (mean 3 seed >= 95.7%) |
| I2 | Macro-F1 cải thiện so với mốc | 5 | 5 | final 0.9641, mốc 0.9210, Δ=+0.0431, s=0.0025 (Δ > s và Δ >= 0.01) |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 96.0% (mốc 88.5%), Snake Weed 96.3% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0482, sau 0.0195 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9672, test 0.9641, chênh 0.0031 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 15.2 ms (ngân sách 100 ms), đo đúng cách (warmup + sync) |

Tổng các ý đã chấm: 20 / 20 (Phần I đạt tối đa 20/20)
```

---

## 7. Phân tích lỗi & Ma trận nhầm lẫn

### 7.1 Hai lớp khó: Chinee Apple và Snake Weed
- **Hiện tượng bài báo gốc:** Cặp *Chinee Apple* và *Snake Weed* dễ bị nhầm lẫn với nhau nhất do cấu trúc tán lá nhỏ, màu xanh lục tương đồng và nền cỏ khô che khuất.
- **Kết quả mô hình F01:** Nhờ **CutMix** và **ConvNeXt**, mô hình học được đặc trưng vân lá và đường gân chi tiết thay vì phụ thuộc hình dáng tổng thể, giúp tăng Recall từ mức $89\%$ (Baseline) lên **$96.0\%$** và **$96.3\%$**.

---

## 8. Hạn chế & Khuyến nghị triển khai

1. **Chia dữ liệu ngẫu nhiên (Random Split):** Việc chia ngẫu nhiên 60/20/20 có thể làm tập Test có sự tương đồng về bối cảnh với tập Train (cùng ngày chụp hoặc cùng vị trí luống đất). Trong triển khai thực tế, khuyến nghị chia theo địa điểm địa lý (Spatial/Site-based split).
2. **Khuyến nghị phần cứng:** Triển khai mô hình `convnext_tiny` dưới định dạng **ONNX Runtime / TensorRT FP16** trên các dòng chip nhúng (như NVIDIA Jetson Orin Nano) để đạt tốc độ xử lý $60\text{ FPS}$ liên tục ngoài thực địa.
