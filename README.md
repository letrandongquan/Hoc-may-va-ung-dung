
## 1.1 Biến đổi cường độ ảnh (Image Inverse Transformation)

chương trình thực hiện biến đổi cường độ ảnh 

### Công nghệ sử dụng:
- `Pillow (PIL)` mở ảnh và xử lý định dạng.
- `NumPy`chuyển đổi và thao tác với ma trận ảnh.
- `Matplotlib` hiển thị ảnh kết quả.

### Mô tả:
1. Mở ảnh gốc và chuyển sang ảnh xám (`grayscale`).
2. Chuyển ảnh thành mảng số (`numpy array`).
3. Tạo ảnh âm bản bằng công thức: `255 - pixel`.
4. Hiển thị ảnh gốc và ảnh âm bản để so sánh.

### giải thích hoạt động:
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

# Mở ảnh và chuyển sang grayscale
img = Image.open('bird.png').convert('L')

# Chuyển ảnh thành mảng số
im_1 = np.asarray(img)

# Tạo ảnh âm bản
im_2 = 255 - im_1

# Tạo ảnh mới và hiển thị
new_img = Image.fromarray(im_2)

img.show()  # Hiển thị ảnh gốc
plt.imshow(im_2, cmap='gray')  
plt.show()


## 1.2 Thay đổi chất lượng ảnh với Power Law (Gamma Correction)
### sinh viên thực hành thay đổi giá trị gamma = 5

chương trình thực hiện biến đổi cường độ ảnh theo quy luật Power Law (Gamma Correction), giúp điều chỉnh độ sáng của ảnh.

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở ảnh và xử lý định dạng.
- `NumPy` để chuyển đổi và thao tác với ma trận ảnh.
- `Matplotlib` để hiển thị ảnh kết quả.

### Mô tả hoạt động:
1. Mở ảnh gốc và chuyển sang ảnh xám (`grayscale`).
2. Chuyển ảnh thành mảng số (`numpy array`) với kiểu số thực để tính toán.
3. Chuẩn hóa mảng pixel về khoảng [0,1].
4. Áp dụng biến đổi Gamma: \( I_{out} = \log(I_{in} + \epsilon) \times \gamma \).
5. Giới hạn giá trị pixel về khoảng [0,255] và chuyển về kiểu số nguyên 8-bit.
6. Tạo ảnh mới từ mảng kết quả.
7. Hiển thị ảnh gốc và ảnh sau biến đổi.


### Giải thích các hoạt động:

# Chuyển ảnh thành mảng số thực để tính toán
im_1 = np.asarray(img)
b1 = im_1.astype(float)
# Chuyển ảnh sang mảng NumPy và đổi sang kiểu float để tính toán chính xác

# Chuẩn hóa pixel về khoảng 0 đến 1
b2 = np.max(b1)
b3 = b1 / b2
# Chia mỗi pixel cho giá trị lớn nhất, chuẩn hóa dữ liệu về [0, 1]

# Áp dụng biến đổi Gamma: log(I_in + epsilon) * gamma
gamma = 5
b2 = np.log(b3 + 1e-5) * gamma
# Biến đổi gamma sử dụng logarit, thêm 1e-5 để tránh log(0)
# Gamma = 5 điều chỉnh độ sáng tối

# Giới hạn giá trị và chuyển về uint8 (0-255)
c1 = np.clip(b2, 0, 255).astype(np.uint8)
# Giới hạn giá trị trong 0-255, chuyển về kiểu uint8

# Tạo ảnh mới từ mảng kết quả
d = Image.fromarray(c1)
# Tạo ảnh mới từ mảng pixel đã biến đổi để hiển thị hoặc lưu trữ


## 1.3 Thay đổi cường độ điểm ảnh với Log Transformation

chương trình thực hiện biến đổi cường độ ảnh theo hàm logarit (Log Transformation), giúp tăng cường các chi tiết vùng tối trong ảnh.

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở ảnh và xử lý định dạng.
- `NumPy` để chuyển đổi và thao tác với ma trận ảnh.
- `Matplotlib` để hiển thị ảnh kết quả.

### Mô tả hoạt động:
1. Mở ảnh gốc và chuyển sang ảnh xám (grayscale).
2. Chuyển ảnh thành mảng số (`numpy array`) với kiểu số thực để tính toán.
3. Tính giá trị lớn nhất trong ảnh để chuẩn hóa.
4. Áp dụng biến đổi Log theo công thức:
   \[
   I_{out} = \frac{c \cdot \log(1 + I_{in})}{\log(1 + I_{max})}
   \]
   trong đó \(c = 128.0\) là hệ số điều chỉnh.
5. Giới hạn giá trị pixel về khoảng [0,255] và chuyển về kiểu số nguyên 8-bit.
6. Tạo ảnh mới từ mảng kết quả.
7. Hiển thị ảnh gốc và ảnh đã biến đổi.

### Giải thích các hoạt động:

# Chuyển ảnh sang mảng số thực để xử lý
im_1 = np.asarray(img)
b1 = im_1.astype(float)

# Tìm giá trị pixel lớn nhất
b2 = np.max(b1)

# Áp dụng biến đổi logarit với hệ số c=128
c = (128.0 * np.log(1 + b1)) / np.log(1 + b2)

# Giới hạn giá trị trong khoảng 0-255 và chuyển về uint8
cl = np.clip(c, 0, 255).astype(np.uint8)

# Tạo ảnh mới từ mảng kết quả
d = Image.fromarray(cl)

## 1.4 Histogram Equalization

Đoạn code dưới đây thực hiện cân bằng histogram cho ảnh xám (grayscale) nhằm cải thiện độ tương phản bằng cách trải đều phổ mức xám.

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở và xử lý ảnh.
- `NumPy` để thao tác trên mảng ảnh.
- `Matplotlib` để hiển thị ảnh.

### Mô tả hoạt động:
1. Mở ảnh gốc và chuyển sang ảnh xám.
2. Chuyển ảnh thành mảng 1 chiều (`flatten`) để tính histogram.
3. Tính histogram và hàm phân phối tích lũy (CDF).
4. Chuẩn hóa CDF để trải đều mức xám từ 0 đến 255.
5. Áp dụng CDF chuẩn hóa lên từng pixel ảnh ban đầu.
6. Tạo ảnh mới từ mảng kết quả.
7. Hiển thị ảnh gốc và ảnh sau cân bằng histogram.

### Giải thích các hoạt động:


# Chuyển ảnh thành mảng numpy và làm phẳng (flatten) để tính histogram
im1 = np.asarray(img)
bl = im1.flatten()

# Tính histogram và các mức bin (256 mức từ 0 đến 255)
hist, bins = np.histogram(bl, 256, [0, 255])

# Tính hàm phân phối tích lũy (CDF)
cdf = hist.cumsum()

# Loại bỏ các giá trị bằng 0 để tránh chia cho 0
cdf_m = np.ma.masked_equal(cdf, 0)

# Chuẩn hóa CDF về khoảng 0-255
num_cdf_m = (cdf_m - cdf_m.min()) * 255
den_cdf_m = (cdf_m.max() - cdf_m.min())
cdf_m = num_cdf_m / den_cdf_m

# Thay các giá trị mask bằng 0 và chuyển về kiểu uint8
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

# Áp dụng CDF chuẩn hóa lên ảnh ban đầu
im2 = cdf[bl]

# Đưa ảnh về dạng 2D ban đầu
im3 = np.reshape(im2, im1.shape)

# Tạo ảnh PIL mới từ mảng kết quả
im4 = Image.fromarray(im3.astype(np.uint8))

## 1.5 Thay đổi ảnh với Contrast Stretching

chương trình thực hiện tăng cường tương phản ảnh bằng phương pháp kéo giãn dải cường độ (Contrast Stretching), giúp tận dụng toàn bộ khoảng cường độ từ 0 đến 255.

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở và xử lý ảnh.
- `NumPy` để thao tác trên mảng ảnh.
- `Matplotlib` để hiển thị ảnh.

### Mô tả hoạt động:
1. Mở ảnh gốc và chuyển sang ảnh xám.
2. Lấy giá trị cường độ nhỏ nhất (`a`) và lớn nhất (`b`) trong ảnh.
3. Thực hiện phép biến đổi tuyến tính kéo giãn các mức xám:
   \[
   I_{out} = 255 \times \frac{I_{in} - a}{b - a}
   \]
4. Chuyển mảng kết quả thành ảnh mới.
5. Hiển thị ảnh gốc và ảnh đã kéo giãn tương phản.

### Giải thích các hoạt động:



# Chuyển ảnh thành mảng numpy
im1 = np.asarray(img)

# Lấy giá trị min và max trong ảnh
a = im1.min()
b = im1.max()
print(a, b)  # In ra giá trị min và max để tham khảo

# Chuyển sang float để tính toán
c = im1.astype(float)

# Áp dụng biến đổi Contrast Stretching
im2 = 255 * (c - a) / (b - a)

# Tạo ảnh mới từ mảng kết quả
im3 = Image.fromarray(im2.astype(np.uint8))



### 1.6 Biến Đổi Fourier
## 1.6.1 Biến đổi ảnh với Fast Fourier Transform (FFT)

chương trình sử dụng phép biến đổi Fourier nhanh (FFT) để chuyển ảnh từ miền không gian sang miền tần số, giúp phân tích cấu trúc tần số của ảnh.

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở và xử lý ảnh.
- `SciPy` để tính toán FFT.
- `NumPy` để thao tác mảng.
- `Matplotlib` để hiển thị ảnh.

### Mô tả hoạt động:
1. Mở ảnh và chuyển sang ảnh xám (grayscale).
2. Chuyển ảnh thành mảng số (`numpy array`).
3. Tính FFT 2 chiều của ảnh với `scipy.fftpack.fft2()`.
4. Lấy giá trị tuyệt đối của FFT (biểu diễn biên độ tần số).
5. Dịch FFT về tâm phổ tần số với `fftshift()` để dễ quan sát.
6. Áp dụng logarit để tăng cường hiển thị phổ tần số.
7. Chuẩn hóa và chuyển kết quả về khoảng 0-255.
8. Tạo ảnh mới từ mảng kết quả.
9. Hiển thị ảnh gốc và ảnh phổ tần số.

### Giải thích các hoạt động:


# Chuyển ảnh thành mảng numpy
im1 = np.asarray(img)

# Tính FFT 2 chiều và lấy giá trị tuyệt đối (biên độ phổ)
c = abs(scipy.fftpack.fft2(im1))

# Dịch phổ về trung tâm (trong miền tần số)
d = scipy.fftpack.fftshift(c)

# Áp dụng logarit để tăng cường hiển thị
d = np.log(1 + d)

# Chuẩn hóa về khoảng 0-255
d = 255 * d / np.max(d)
d = d.astype(np.uint8)

# Tạo ảnh mới từ phổ tần số
im3 = Image.fromarray(d)


## 1.6.2 Lọc ảnh trong miền tần suất

chương trình thực hiện lọc ảnh trong miền tần số bằng cách áp dụng bộ lọc Butterworth thấp tần (low-pass filter) lên phổ tần số của ảnh, giúp làm mờ ảnh và giảm nhiễu cao tần.

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở và xử lý ảnh.
- `SciPy` để tính toán FFT và IFFT.
- `NumPy` để thao tác mảng.
- `Matplotlib` để hiển thị ảnh.

### Mô tả hoạt động:
1. Mở ảnh và chuyển sang ảnh xám.
2. Chuyển ảnh thành mảng số (`numpy array`).
3. Tính FFT 2 chiều và dịch phổ về tâm tần số.
4. Tạo bộ lọc Butterworth thấp tần với ngưỡng cắt `d_0`.
5. Áp dụng bộ lọc lên phổ tần số.
6. Tính IFFT để chuyển ngược về miền không gian.
7. Chuyển kết quả thành ảnh mới.
8. Hiển thị ảnh gốc và ảnh đã lọc.

### Giải thích các hoạt động:


# Chuyển ảnh thành mảng numpy
im1 = np.asarray(img)

# Tính FFT 2 chiều và lấy biên độ phổ
c = abs(scipy.fftpack.fft2(im1))

# Dịch phổ về trung tâm
d = scipy.fftpack.fftshift(c)

M, N = d.shape
H = np.ones((M, N))

center1, center2 = M / 2, N / 2
d_0 = 30.0  # Ngưỡng cắt (cutoff frequency)
t1 = 1      # Tham số bậc bộ lọc
t2 = 2 * t1

# Tạo bộ lọc Butterworth thấp tần
for i in range(1, M):
    for j in range(1, N):
        rl = (i - center1)**2 + (j - center2)**2
        r = math.sqrt(rl)
        if r > d_0:
            H[i, j] = 1 / (1 + (r / d_0)**t1)

H = H.astype(float)

# Áp dụng bộ lọc lên phổ tần số
con = d * H

# Tính IFFT để đưa ảnh về miền không gian
e = abs(scipy.fftpack.ifft2(con))
e = e.astype(float)

# Tạo ảnh mới từ mảng kết quả
im3 = Image.fromarray(e)


### Bài tập 
# 1. Viết chương trình tạo menu cho phép người dùng chọn các phương pháp biến đổi ảnh như
sau:
- Image inverse transformation
- Gamma-Correction
- Log Transformation
- Histogram equalization
- Contrast Stretching
Khi người dùng ấn phím I, G, L, H, C thì chương trình sẽ thực hiện hàm tương ứng cho các hình trong thư mục exercise. Lưu và hiển thị các ảnh đã biến đổi.

Chương trình cho phép người dùng chọn một trong các phương pháp biến đổi ảnh sau để xử lý hàng loạt các ảnh trong thư mục `exercise`. Kết quả được lưu và hiển thị.

---

## Các phương pháp biến đổi ảnh

- **Image inverse transformation (I)**: Biến đổi ảnh âm bản (đảo ngược cường độ điểm ảnh).
- **Gamma Correction (G)**: Điều chỉnh độ sáng ảnh theo luật hàm mũ (power-law).
- **Log Transformation (L)**: Biến đổi cường độ điểm ảnh theo hàm logarit.
- **Histogram Equalization (H)**: Cân bằng histogram ảnh xám để cải thiện độ tương phản.
- **Contrast Stretching (C)**: Kéo dài dải cường độ ảnh để tăng độ tương phản.

---

## Công nghệ sử dụng

- `OpenCV` (`cv2`) để đọc, ghi và xử lý ảnh.
- `NumPy` để xử lý mảng số và tính toán.
- `Matplotlib` để hiển thị ảnh.

---

## Hướng dẫn sử dụng

1. Đặt các ảnh cần xử lý vào thư mục `exercise` (hỗ trợ định dạng: `.png`, `.jpg`, `.jpeg`, `.bmp`).
2. Chạy chương trình Python.
3. Nhập phím tương ứng với phương pháp biến đổi ảnh bạn muốn thực hiện (I/G/L/H/C).
4. Chương trình sẽ xử lý tất cả ảnh trong thư mục `exercise`, lưu kết quả trong thư mục `output` và hiển thị từng ảnh đã biến đổi.

---

## giải thích hoạt động

def image_inverse(img):
    # Trả về ảnh âm bản: 255 - pixel
    return 255 - img

def gamma_correction(img, gamma=2.2):
    # Áp dụng biến đổi gamma (power-law)
    normalized = img / 255.0
    corrected = np.power(normalized, 1/gamma)
    return np.uint8(corrected * 255)

def log_transformation(img):
    # Biến đổi cường độ theo hàm logarit
    c = 255 / np.log(1 + np.max(img))
    log_image = c * np.log(1 + img.astype(np.float32))
    return np.uint8(log_image)

def histogram_equalization(img):
    # Cân bằng histogram ảnh xám
    return cv2.equalizeHist(img)

def contrast_stretching(img):
    # Kéo dài dải cường độ ảnh để tăng độ tương phản
    a, b = np.min(img), np.max(img)
    stretched = (img - a) * (255.0 / (b - a))
    return np.uint8(stretched)


# 2. Viết chương trình tạo menu cho phép người dùng chọn các phương pháp biến đổi ảnh như
sau:
- Fast Fourier
- Butterworth Lowpass Filter
- Butterworth Highpass Filter
Khi người dùng ấn phím F, L, H thì chương trình sẽ thực hiện hàm tương ứng cho các hình trong thư mục exercise. Lưu và hiển thị các ảnh đã biến đổi.

Chương trình cho phép người dùng chọn một trong các phương pháp biến đổi ảnh miền tần số để xử lý hàng loạt các ảnh trong thư mục `exercise`. Kết quả được lưu và hiển thị.

---

## Các phương pháp biến đổi ảnh miền tần số

- **Fast Fourier Transform (F)**: Chuyển đổi ảnh sang miền tần số, hiển thị phổ biên độ.
- **Butterworth Lowpass Filter (L)**: Lọc thông thấp Butterworth trong miền tần số để làm mờ ảnh.
- **Butterworth Highpass Filter (H)**: Lọc thông cao Butterworth trong miền tần số để làm nổi bật biên.

---

## Công nghệ sử dụng

- `OpenCV` (`cv2`) để đọc, ghi và xử lý ảnh.
- `NumPy` để xử lý mảng số và tính toán.
- `Matplotlib` để hiển thị ảnh.



## giải thích hoạt động

def apply_fft(img):
    # Tính biến đổi Fourier nhanh (FFT) và trả về phổ biên độ ảnh
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    return np.uint8(magnitude / magnitude.max() * 255)

def butterworth_lowpass_filter(img, d0=30, n=2):
    # Lọc thông thấp Butterworth với ngưỡng cắt d0 và bậc n
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / d0)**(2 * n))

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    result = np.abs(img_back)
    return np.uint8(result / result.max() * 255)

def butterworth_highpass_filter(img, d0=30, n=2):
    # Lọc thông cao Butterworth với ngưỡng cắt d0 và bậc n
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    result = np.abs(img_back)
    return np.uint8(result / result.max() * 255)

### 3. Viết chương trình thay đổi thứ tự màu RGB của ảnh trong thư mục exercise và sử dụng ngẫu nhiên một trong các phép biến đổi ảnh trong câu 1. Lưu và hiển thị ảnh đã biến đổi.
# Chương trình biến đổi ảnh với đảo thứ tự kênh màu RGB và áp dụng ngẫu nhiên phép biến đổi ảnh

Chương trình thực hiện các bước xử lý sau với tất cả ảnh trong thư mục `exercise`:
- Đảo ngẫu nhiên thứ tự các kênh màu RGB của ảnh.
- Áp dụng ngẫu nhiên một trong các phép biến đổi ảnh phổ biến (Inverse, Gamma Correction, Log Transformation, Histogram Equalization, Contrast Stretching).
- Hiển thị đồng thời ảnh gốc, ảnh sau đảo kênh RGB và ảnh sau biến đổi.
- Lưu ảnh đã biến đổi (nếu cần có thể thêm tính năng lưu).

---

## Các phương pháp biến đổi ảnh được hỗ trợ

1. **Image Inverse Transformation**: Đảo ngược cường độ điểm ảnh.
2. **Gamma Correction**: Hiệu chỉnh gamma với hệ số gamma ngẫu nhiên.
3. **Log Transformation**: Biến đổi logarithmic tăng cường chi tiết vùng tối.
4. **Histogram Equalization**: Cân bằng histogram để tăng tương phản.
5. **Contrast Stretching**: Kéo giãn dải cường độ ảnh.

---

## Công nghệ sử dụng

- `PIL` (Pillow) và `imageio` để đọc và xử lý ảnh.
- `NumPy` để xử lý mảng ảnh và các phép toán số học.
- `Matplotlib` để hiển thị ảnh và kết quả.
- `random` để chọn ngẫu nhiên phép biến đổi và trộn kênh màu.

---

## giải thích chương trình

- `shuffle_rgb_channels(img_array)`: Đảo trộn ngẫu nhiên các kênh R, G, B của ảnh.
- `image_inverse(img_array)`: Đảo ngược điểm ảnh.
- `gamma_correction(img_array, gamma)`: Áp dụng hiệu chỉnh gamma với hệ số gamma tùy chọn.
- `log_transform(img_array)`: Biến đổi logarithmic cho ảnh.
- `histogram_equalization(img_array)`: Cân bằng histogram cho ảnh màu hoặc ảnh xám.
- `contrast_stretching(img_array)`: Kéo giãn dải tương phản.
- `apply_random_transform(img_array)`: Lựa chọn ngẫu nhiên một trong các phép biến đổi và áp dụng lên ảnh.
- `process_image(filepath)`: Thực hiện toàn bộ quy trình cho từng ảnh, hiển thị kết quả.
- `main()`: Lặp qua toàn bộ ảnh trong thư mục `exercise` và xử lý từng ảnh.


### 4. Viết chương trình thay đổi thứ tự màu RGB của ảnh trong thư mục exercise và sử dụng ngẫu nhiên một trong các phép biến đổi ảnh trong câu 2. Nếu ngẫu nhiên là phép Butterworth
Lowpass thì chọn thêm Min Filter để lọc ảnh. Nếu ngẫu nhiên là phép Butterworth Highpass thì chọn thêm Max Filter để lọc ảnh. Lưu và hiên thị ảnh đã biến đối.
## 4. RGB Channel Shuffle và Bộ lọc Butterworth kèm Min/Max Filter

Chương trình thực hiện biến đổi ảnh màu bằng cách thay đổi thứ tự các kênh màu RGB ngẫu nhiên, sau đó áp dụng một trong các phép biến đổi miền tần số gồm:

- Biến đổi nghịch đảo ảnh (Inverse)
- Hiệu chỉnh Gamma (Gamma Correction)
- Biến đổi Logarithm (Log Transform)
- Bộ lọc Butterworth Lowpass kết hợp với bộ lọc Min (lọc xói mòn)
- Bộ lọc Butterworth Highpass kết hợp với bộ lọc Max (lọc giãn nở)

### Công nghệ sử dụng:
- `Pillow (PIL)` để mở và xử lý ảnh.
- `NumPy` để xử lý mảng ảnh.
- `SciPy` để thực hiện biến đổi Fourier và các bộ lọc xói mòn, giãn nở.
- `Matplotlib` để hiển thị ảnh.
- `imageio` để đọc ảnh.

### Mô tả hoạt động:

1. Đọc ảnh màu từ thư mục `exercise`.
2. Đảo ngẫu nhiên thứ tự các kênh màu RGB.
3. Chọn ngẫu nhiên một phép biến đổi trong danh sách (Inverse, Gamma, Log, Butterworth Lowpass, Butterworth Highpass).
4. Nếu chọn bộ lọc Butterworth Lowpass thì áp dụng thêm bộ lọc Min (xói mòn) để loại bỏ nhiễu.
5. Nếu chọn bộ lọc Butterworth Highpass thì áp dụng thêm bộ lọc Max (giãn nở) để tăng cường biên.
6. Hiển thị đồng thời ảnh gốc, ảnh shuffle RGB và ảnh sau khi biến đổi.
7. Lưu ảnh kết quả vào thư mục `output`.

### Giải thích hoạt động

# Đọc ảnh màu và shuffle kênh RGB
img = iio.imread('bird.png')
channels = [img[:,:,i] for i in range(3)]
random.shuffle(channels)
shuffled_img = np.dstack(channels)

# Biến đổi Butterworth Lowpass + Min Filter
gray = np.mean(shuffled_img, axis=2)
f = scipy.fftpack.fft2(gray)
fshift = scipy.fftpack.fftshift(f)
rows, cols = gray.shape
crow, ccol = rows//2, cols//2
D0, n = 30, 2
H = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        D = math.sqrt((i-crow)**2 + (j-ccol)**2)
        H[i,j] = 1 / (1 + (D/D0)**(2*n))
filtered = fshift * H
f_ishift = scipy.fftpack.ifftshift(filtered)
img_back = np.abs(scipy.fftpack.ifft2(f_ishift))
img_back = np.uint8(np.clip(img_back, 0, 255))
img_back = scipy.ndimage.grey_erosion(img_back, size=(3,3))  # Min filter
transformed = np.dstack([img_back]*3)

# Biến đổi Butterworth Highpass + Max Filter
# Tương tự nhưng với hàm truyền H thay đổi và dùng grey_dilation (Max filter)


# Hoc-may-va-ung-dung
