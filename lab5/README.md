### Nhập Môn Xử Lý Ảnh Số - Lab 5
### xác định đối tượng trong ảnh
- Sinh viên thực hiện: Lê Trần Đông Quân MSSV: 2374802010414
- Môn học: Nhập môn xử lý ảnh số
- Giảng viên: Đỗ Hữu Quân
## Giới thiệu
Xác định đối tượng trong ảnh là gán nhãn để phân biệt các đối tượng khác nhau trong ảnh . Trong 1 ảnh đã được gán nhãn, tất cả các pixel của một đối tượng có giá trị như nhau.các chương trình sẽ học bao gồm:
- chương trình gán nhãn cho phân vùng ảnh
- chương trình phân vùng ảnh theo region
- chương trình thay đổi ảnh
- image matching

## Công nghệ sử dụng
- Pillow (PIL): Đọc, chuyển đổi, và lưu ảnh
- NumPy: xử lý ảnh dưới dạng mảng số học
- ImageIO: Đọc và ghi file ảnh với định dạng hiện đại
- skimage.morphology.label: Gán nhãn các vùng liên thông trong ảnh nhị phân.
- skimage.measure.regionprops: Trích xuất các thuộc tính hình học (diện tích, tâm, bounding box,...) từ các vùng đã gán nhãn.
- Matplotlib: Hiển thị ảnh và vẽ biểu đồ trực quan.
- Matplotlib.patches: Vẽ hình chữ nhật, hình tròn,...đánh dấu các đối tượng trên ảnh.
- threshold_otsu: Phân ngưỡng ảnh nhị phân tự động theo phương pháp Otsu.
- corner_harris: Phát hiện các điểm góc (Harris corners) trong ảnh.
- rgb2gray: Chuyển đổi ảnh màu RGB sang ảnh xám.
- OpenCV : Thư viện xử lý ảnh thời gian thực.
## Chi tiết các phương pháp & công thức
## Cài Đặt thư viện
pip install opencv-python
## 2. Viết chương trình gán nhãn ảnh
## 2.1 Gán nhãn ảnh
# Mục đích:
- Xác định và gán nhãn cho các vùng liên thông, mỗi vùng đối tượng riêng biệt sẽ được gán một giá trị nhãn khác nhau. giúp đếm số lượng đối tượng trong ảnh, trích xuất đặc trưng hình học
# Công thức toán học
1. phương pháp otsu
- b(x, y) = {
    1, nếu a(x, y) > T
    0, nếu a(x, y) ≤ T
}
- a(x,y): giá trị pixel ảnh xám.
T
T: ngưỡng Otsu tự động tính toán.
2. gán nhãn vùng liên thông
c(x,y)=i,i=1,2,...,N
N: tổng số vùng liên thông.
c(x,y): ảnh gán nhãn.
3. trích xuất thuộc tính
Centroid: trọng tâm
- centroid_x = (1 / A) * ∑_{(x, y) ∈ R} x  
- centroid_y = (1 / A) * ∑_{(x, y) ∈ R} y
Area: diện tích: số pixel thuộc vùng.
Bouding box: khung chữ nhật nhỏ nhất chứa toàn bộ vùng.
# ví dụ:
giả sử có một ảnh nhị phân, có 2 vùng liên thông được xác định , vùng 1 ( ở trên) gán nhãn là 1 và vùng 2(ở dưới): gán nhãn là 2
```python
0 0 1 1 0
0 0 1 1 0
0 0 0 0 0
0 1 1 0 0
0 1 1 0 0
```
sau khi đã gán nhãn:
```python
0 0 1 1 0
0 0 1 1 0
0 0 0 0 0
0 2 2 0 0
0 2 2 0 0
```
area vùng 1 bằng 4 pixel, Bounding box của vùng 2 từ (3, 1) đến (5, 3), centroid được tính theo toạ độ trung bình của các điểm trong vùng.

# Code chính: 
```python
a = np.asarray(data)
thres = threshold_otsu(a)
b = a > thres
c = label(b)
c1 = Image.fromarray((c * 255 / c.max()).astype(np.uint8))
iio.imsave('label_output.jpg', c1)
properties = ['Area', 'Centroid', 'BoundingBox']
d = regionprops(c)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(c, cmap='YlOrRd')
for i in d:
    lr, lc, ur, uc = i['BoundingBox']
    rec_width = uc - lc
    rec_height = ur - lr
    rect = mpatches.Rectangle((lc, lr), rec_width, rec_height,
                              fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

plt.show()
```
## 2.2 Dò tìm cạnh theo chiều dọc
# Mục đích:
Xác định các ranh giới dọc (vertical edges) giữa các vùng sáng,tối trong ảnh xám, nhằm nhận diện được hình dạng của vật thể
# Công thức toán học
s(x,y)=∣a(x,y)−a(x,y+1)∣
với :
a(x,y): ảnh gốc 
s(x,y): ảnh biên theo chiều dọc tại vị trí của
(x,y)
-Hàm nd.shift(a, (0, 1), order=0) dịch toàn bộ ảnh 1 pixel sang trái (theo trục y)
-lấy giá trị tuyệt đối của hiệu để phát hiện sự thay đổi
# ví dụ:
[100, 105, 150, 150, 155]
dò cạnh dọc:
|100 - 105| = 5
|105 - 150| = 45
|150 - 150| = 0
|150 - 155| = 5
kết quả là [5, 45, 0, 5] vị trí có độ chênh lớn sẽ hiển thị rõ hơn(cạnh sáng)

# Code chính: 
```python
a = np.asarray(data)
bmg = abs(a - nd.shift(a, (0, 1), order=0))
```
## 2.3 Dò tìm cạnh với Sobel Filter
# Mục đích:
Phát hiện các cạnh biên trong ảnh theo cả chiều dọc và ngang. Xác định ranh giới rõ ràng giữa các vật thể và nền, dùng trong phân đoạn, nhận diện  ảnh
# Công thức toán học
toán tử sobel, là các kernel tích chập (convolution) để tính xấp xỉ đạo hàm nhật nhất. theo từng chiều
-chiều ngang (Gx):
-chiều dọc (Gy):
- Gx = [ [-1,  0, +1],
       [-2,  0, +2],
       [-1,  0, +1] ]

- sobel theo chiều dọc (Gy):
- Gy = [ [-1, -2, -1],
       [  0,  0,  0],
       [+1, +2, +1] ]

-Công thức tính độ lớn gradient(gradient Magnitude):
       Gradient = |Gx| + |Gy|
 
# ví dụ:
ta có ảnh biên rõ ràng, sobel sẽ cho ra kết quả:
vùng đồng nhất không biên : giá trị gần 0
vùng có ranh giới thay đổi mạnh: giá trị lớn( sáng)
-các vị trí có giá trị lớn là nơi có thay đổi mạnh về độ sángnhư các cạnh của vật thể

# Code chính: 
```python
a = np.asarray(data)
gx = nd.sobel(a, axis=0)  
gy = nd.sobel(a, axis=1)  
bmg = np.abs(gx) + np.abs(gy)
```
## 2.4 Xác định góc của đối tượng
# Mục đích:
- phát hiện các điểm góc trong ảnh, những vị trí có sự thay đổi rõ rệt về hướng cả hai trục x và y
giúp nhận diện vật thể, theo dõi chuyển động,...
# Công thức toán học
đạo hàm bật nhất theo chiều x và y:
- Ix = ∂I/∂x  ≈ sobel_x(I)
- Iy = ∂I/∂y  ≈ sobel_y(I)

ma trận cấu trúc harris (structure matrix C)

- C = [ Ix2   Ixy
      Ixy   Iy2 ]
Ix2 = gaussian_blur(Ix2)
Iy2 = gaussian_blur(Iy2)
Ixy = gaussian_blur(Ixy)
-điểm phản hồi r
 R = det(C) - α * (trace(C))^2
 với
det(C) = Ix2 * Iy2 - Ixy^2
trace(C) = Ix2 + Iy2
α ∈ [0.04, 0.06], ví dụ α = 0.04 hoặc α = 0.2
ý nghĩa của R
R > 0: vùng có góc rõ
R xấp sỉ bằng 0: vùng biên hoặc phẳng
R < 0: vùng không đặc trưng
# Cách thức hoạt động:
- khi áp dụng thuật toán harris vào một ảnh có hình vuông: các điểm ở góc của hình vuông sẽ có giá trị R lớn (sáng), vùng màu đều và đồng nhất thì R xấp sỉ bằng 0 , các cạnh có R thấp hơn góc nhưng > nền 
# Code chính: 
```python
def Harris(indata, alpha=0.2):
    x = nd.sobel(indata, 0)  
    y = nd.sobel(indata, 1)  
    x1 = x ** 2
    y1 = y ** 2
    xy = x * y

    x1 = nd.gaussian_filter(x1, 3)
    y1 = nd.gaussian_filter(y1, 3)
    xy = nd.gaussian_filter(xy, 3)

    detC = x1 * y1 - xy**2
    trC = x1 + y1
    R = detC - alpha * (trC**2)

    return R
data = Image.open('geometric.png').convert('L')
data_array = np.array(data)
bmg = Harris(data_array)
```
## 2.5 Dò tìm hình dạng cụ thể trong ảnh với Hough Transform
## 2.5.1 Dò tìm đường thẳng trong ảnh
# Mục đích:
- chương trình giúp hát hiện các đường thẳng có mặt trong ảnh nhị phân hoặc ảnh biên, sử dụng để nhận diện làn đường, phân tích biên dạng, đo kích thước , khoảng cách hoặc phát hiện cấu trúc tuyến tính.
# Công thức toán học
Biến đổi không gian ảnh sang không gian Hough:
ρ=x⋅cos(θ)+y⋅sin(θ)
ρ là khoảng cách từ gốc tọa độ đến đường thẳng.
θ là góc của pháp tuyến (0–180°).
Bộ đếm tích lũy (Accumulator Array):
accumulator[ρ,θ]+=1
# Cáh thức hoạt động
# ví dụ:
- Ảnh đầu vào là ảnh 256 X 256 , chỉ có một điểm duy nhất sáng ở (128, 128)
- Biến đổi Hough sẽ tạo ra một đường sin trong không gian Hough, một điểm tương ứng với vô số đường đi qua nó
- ma trận kết quả sẽ có dạng hình sin trong mặt phẳng(ρ, θ)

# Code chính: 
```python
def LineHough(data, gamma):
    V, H = data.shape
    R = int(np.sqrt(V * V + H * H))
    ho = np.zeros((R, 90), float)  
    w = data + 0
    ok = 1
    theta = np.arange(90) / 180.0 * np.pi
    tp = np.arange(90).astype(float)

    while ok:
        mx = w.max()
        if mx < gamma:
            ok = 0
        else:
            v, h = divmod(w.argmax(), H)
            y = V - v
            x = h
            rh = x * np.cos(theta) + y * np.sin(theta)
            for i in range(len(rh)):
                if 0 <= rh[i] < R and 0 <= tp[i] < 90:
                    ho[int(rh[i]), int(tp[i])] += mx
            w[v, h] = 0
    return ho
data = np.zeros((256, 256))
data[128, 128] = 1

bmg = LineHough(data, 0.5)
```
## 2.5.2 Dò tìm đường tròn trong ảnh
# Mục đích:
Phát hiện các đối tượng có dạng hình tròn trong ảnh như con mắt, bánh xe, quả bóng,...dùng để nhận diện vật thể
# Công thức toán học
corner harris, phát hiện điểm đặc trưng
R=det(C)−k*(trace(C))^ 2
C: ma trận cấu trúc dựa trên đạo hàm theo x và y

k: hệ số nhạy ( k = 0.001)

R: mức phản hồi Harris – càng lớn thì càng có khả năng là góc
 
# cách thức hoạt đông
# ví dụ:
-cho ảnh đầu vào là bird.png, hình con chim có đôi mắt là hình tròn, áp dụng corner_harris, các điểm có giá trị cao là nơi có biến đổi mạnh về hướng nên có thể nằm trên cạnh của hình tròn
# Code chính: 
```python
image_gray = rgb2gray(data)
coordinate = corner_harris(image_gray, k=0.001)

plt.figure(figsize=(20, 10))
```
## 2.6 Image matching
# Mục đích:
Chương trình sẽ tìm và ghép nối các điểm tương đồng giữa hai ảnh khác nhau của cùng một cảnh hay vật thể, được ứng dụng trong ghép ảnh toàn cảnh, theo dõi chuyển động hoặc so sánh, nhận diện đối tượng
# Công thức toán học và nguyên lý: 
Phát hiện điểm đặc trưng (Corner Harris)
R=det(C)−k*(trace(C))^ 2
với
C: ma trận cấu trúc tại mỗi điểm ảnh

k: hệ số nhạy k trong khoảng 0.04-0.06

R: mức phản hồi Harris – càng lớn thì càng có khả năng là các điểm đặc trưng
Mô tả đặc trưng (Descriptors)
desc=(x−μ)/ σ+ϵ
- tại mỗi điểm đặc trưng, lấy một patch kích thước 11 x 11 xung quanh 
- ​So khớp đặc trưng (Matching Descriptors):
- dựa vào ảnh thứ nhất, tìm đặc trưng ảnh gần nhất hoặc gần nhì từ ảnh thứ hai dựa trên Euclidean
- Áp dụng tỷ lệ kiểm định Lowe (Ratio Test):
 best/second < ratio
 nếu điều kiện đúng ratio = 0.75, ta coi như matching hợp lệ
# ví dụ:
- ảnh 1 và ảnh 2 có vật thể đối tượng giống nhau nhưng góc chụp khác nhau, thì cac 1diem639 giống nhau như, mái nhà, cửa, cây,... sẽ được nối giữa hai ảnh
- vòng tròn xanh lá là điểm đặc trưng ảnh 1, màu xanh dương là điểm đặc trưng ảnh 2
# Code chính: 
```python
def harris_corners(img_gray, threshold=0.01):
    img_float = np.float32(img_gray)
    dst = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    points = np.argwhere(dst > threshold * dst.max())
    return [tuple(p[::-1]) for p in points]  

def get_patch(img, point, patch_size=11):
    x, y = point
    r = patch_size // 2
    if x - r < 0 or y - r < 0 or x + r >= img.shape[1] or y + r >= img.shape[0]:
        return None
    return img[y - r:y + r + 1, x - r:x + r + 1]

def compute_descriptors(img_gray, keypoints, patch_size=11):
    descriptors = []
    valid_pts = []
    for pt in keypoints:
        patch = get_patch(img_gray, pt, patch_size)
        if patch is not None:
            desc = patch.flatten().astype(np.float32)
            desc = (desc - np.mean(desc)) / (np.std(desc) + 1e-8)
            descriptors.append(desc)
            valid_pts.append(pt)
    return valid_pts, descriptors

def match_descriptors_ratio_test(desc1, pts1, desc2, pts2, ratio=0.75):
    matches = []
    desc2 = np.array(desc2)
    for i, d1 in enumerate(desc1):
        dists = np.linalg.norm(desc2 - d1, axis=1)
        if len(dists) < 2:
            continue
        idx = np.argsort(dists)
        best, second = dists[idx[0]], dists[idx[1]]
        if best / (second + 1e-8) < ratio:
            matches.append((pts1[i], pts2[idx[0]]))
    return matches


def draw_matches_gray(img1_gray, img2_gray, matches):
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    canvas = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1] = img1_gray
    canvas[:h2, w1:] = img2_gray

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas, cmap='gray')
    for (x1, y1), (x2, y2) in matches:
        ax.plot([x1, x2 + w1], [y1, y2], 'r-', linewidth=0.5)
        ax.add_patch(mpatches.Circle((x1, y1), radius=3, color='lime', fill=True))
        ax.add_patch(mpatches.Circle((x2 + w1, y2), radius=3, color='blue', fill=True))


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    img1_gray = np.array(Image.open('bali_1.jpg').convert('L'))
    img2_gray = np.array(Image.open('bali_2.jpg').convert('L'))
    kp1 = harris_corners(img1_gray)
    kp2 = harris_corners(img2_gray)
    kp1, desc1 = compute_descriptors(img1_gray, kp1)
    kp2, desc2 = compute_descriptors(img2_gray, kp2)
    matches = match_descriptors_ratio_test(desc1, kp1, desc2, kp2)
    draw_matches_gray(img1_gray, img2_gray, matches)
```
### Tài liệu tham khảo
- Scikit-image: Image processing in Python
- https://medium.com/@sahilutekar.su/revolutionizing-image-matching-with-keynet-affnet-and-hardnet-in-kornia-26d005b3c24
- Slide bài giảng Nhập môn Xử lý ảnh số - Văn Lang University
