### Nhập Môn Xử Lý Ảnh Số - Lab 4
## PHÂN VÙNG ẢNH
Sinh viên thực hiện: Lê Trần Đông Quân MSSV: 2374802010414
Môn học: Nhập môn xử lý ảnh số
Giảng viên: Đỗ Hữu Quân

## Giới thiệu
Phân vùng ảnh là quá trình chia ảnh thành nhiều vùng có chung đặc tính Bài lab này mục đích hướng dẫn viết các chương trình phân vùng ảnh và thay đổi ảnh, các chương trình sẽ học bao gồm:
- Phân vùng theo histogram : chọn ngưỡng phân vùng dựa trên đặc trưng của ảnh, tách nền và vật thể
- Phân vùng theo region : tách vùng phức tạp, xử lý ảnh có nhiều vật thể, mức sáng khác nhau
- Biến đổi đối tượng trong ảnh : co, giãn (erosion, dilation), đóng, mở(closing, opening),...

## Công nghệ sử dụng
- Pillow (PIL): Đọc, chuyển đổi, và lưu ảnh
- NumPy: xử lý ảnh dưới dạng mảng số học
- ImageIO: Đọc và ghi file ảnh với định dạng hiện đại
- Scipy.ndimage: Cung cấp các hàm xử lý ảnh nâng cao
- Matplotlib: Hiển thị ảnh trực quan
- threshold_otsu: phân ngưỡng ảnh nhị phân tự động
- Scikit-image (adaptive) : phân ngưỡng ảnh không đồng đều ánh sáng.
- SciPy Labeling : Gán nhãn các vùng liên thông trong ảnh nhị phân
- OpenCV : xử lý ảnh thời gian thực
## Chi tiết các phương pháp & công thức
## Cài Đặt thư viện
pip install opencv-python
# 2.1 phân vùng theo histogram
# 2.1.1 phương pháp otsu
#Mục đích:
- thuật toán otsu sử dụng để Tính ngưỡng một cách tự động dựa vào giá trị điểm ảnh của ảnh đầu vào nhằm thay thế cho việc sử dụng ngưỡng cố định 

Công thức toán học:
σ_b²(t) = ω_0(t) * ω_1(t) * [μ_0(t) − μ_1(t)]²
ω₀(t), ω₁(t): xác suất xuất hiện của nền và vật thể tại ngưỡng t
μ₀(t), μ₁(t): giá trị trung bình mức xám của nền và vật thể tại ngưỡng t

Ví dụ:
Cho ảnh xám có giá trị pixel trong khoảng 0–255.
Nếu Otsu tính được ngưỡng là t = 123, thì:

Pixel <= 123 được coi là nền (0)
Pixel > 123 được coi là đối tượng (1)

Code chính: 
```python
a = np.asarray(data)
thres = threshold_otsu(a)
b = a > thres
b = Image.fromarray(b)
```
## 2.1.2 phương pháp adaptive thresholding
# Mục đích:
-Cải tiến phân vùng chính xác hơn Otsu. Chia ảnh thành nhiều ảnh nhỏ và tính threshold cho từng ảnh nhỏ 
-giải quyết tốt ảnh có ánh sáng không đều, vùng sáng tối xen kẽ.
# Công thức toán học:
s(x, y) =     1 nếu r(x, y) > T(x, y)
              0 nếu r(x, y) ≤ T(x, y)
r(x, y): giá trị pixel tại vị trí (x, y)
T(x, y): ngưỡng được tính riêng cho mỗi vùng xung quanh điểm (x, y)
s(x, y): giá trị pixel nhị phân mới (0 hoặc 1)
# Ví dụ:
cửa sổ size = 39 và offset = 10, ngưỡng tại mỗi điểm được tính bằng:
giá trị trung bình vùng lân cận – 10
Nếu pixel tại (50, 60) có giá trị 128, còn ngưỡng cục bộ tại đó là 120 → pixel này được gán là 1 (foreground).
# Code chính: 
```python
a = np.asarray(data)
b = threshold_local(a, 39, offset=10)
b = Image.fromarray(b)
```
## 2.2 phân vùng theo region
# Mục đích:
- Một region là một nhóm các pixel có cùng thuộc tính
- sử dụng tách các đối tượng dính liền với nhau
- phân vùng ảnh thành các vùng liên thông dựa vào khoảng cách, cấu trúc và biên
- kết hợp các phương pháp như otsu, erosion, distance transform, watershed để phân vùng chính xác.
# Công thức toán học:
# Công thức Otsu:
σ_b²(t) = ω₀(t) · ω₁(t) · [μ₀(t) − μ₁(t)]²
ω₀(t), ω₁(t): xác suất xuất hiện của nền và vật thể tại ngưỡng t
μ₀(t), μ₁(t): giá trị trung bình mức xám của nền và vật thể tại ngưỡng t
# công thức erosion:
A ⊖ B = { z ∣ B_z ⊆ A }

A  : ảnh nhị phân gốc  
B  : structuring element (phần tử cấu trúc)  
B_z: structuring element B được dịch đến vị trí z  
⊆  : biểu thị toàn bộ B_z nằm bên trong A  
# Công thức Distance Transform
D(x, y) = min_{(x', y') ∈ biên} sqrt((x − x')² + (y − y')²)
D(x, y): giá trị ảnh biến đổi khoảng cách tại điểm ảnh (x,y).
(x', y') thuộc biên: tập các điểm thuộc biên (boundary) trong ảnh nhị phân.
# Watershed
Flood(t)={(x,y)∣H(x,y)≤t}
H(x,y): ảnh địa hình
# Ví dụ:
- ảnh fruit.jpg có nhiều loại trái cây dính liền nhau :
- Otsu để nhị phân hóa
- Erode để làm nhỏ lại và tách rời vật thể
- Distance transform xác định tâm vùng
- Watershed tách vùng dựa trên các tâm đã xác định
# Code chính:
```python
a = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
thresh, b1 = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
b2 = cv2.erode(b1, None, iterations = 2)
dist_trans = cv2.distanceTransform(b2, 2, 3)
thresh, dt = cv2.threshold(dist_trans, 1, 255, cv2.THRESH_BINARY)
labelled, ncc = label(dt)
cv2.watershed(data, labelled)
b = Image.fromarray(labelled)
```
## 2.3 biến đổi đối tượng trong ảnh
## 2.3.1 sử dụng binary_dilation
# Mục đích:
- Điều chỉnh độ sáng tổng thể của ảnh bằng cách thay đổi phân bố giá trị pixel.
- Nếu gamma < 1, ảnh sẽ sáng hơn; nếu gamma > 1, ảnh sẽ tối hơn.
Công thức toán học:
-s = c⋅r ^ γ
-γ: hệ số gamma
-c: hệ số chuẩn hóa (thường là 255 nếu chuẩn hóa pixel từ 0–1)
-r: giá trị pixel đã chuẩn hóa (trong khoảng 0–1)
-s: giá trị pixel sau biến đổi
 
# Ví dụ:
Cho gamma = 0.5, pixel gốc r = 100:
s = 255 *(100 /255)^ 0.5 ≈ 159
# Code chính:
```python
data_np = np.array(data)
binary_img = data_np > 128
b = nd.binary_dilation(binary_img, iterations=50)
c = Image.fromarray((b * 255).astype(np.uint8))
```
## 2.3.2 sử dụng binary_opening
# Mục đích:
-Loại bỏ nhiễu nhỏ, đặc biệt là các đốm trắng nhỏ trong ảnh nhị phân.
- phép mở ảnh nhị phân gồm 2 bước Erosion(co), làm mỏng và loại bỏ điểm nhỏ và dilation(giãn) khôi phục lại hình dạng gốc sau khi loại bỏ nhiễu.
-làm sạch biên, làm mượt đối tượng, lọc các vật thể nhỏ.
# Nguyên lý:
- Ảnh được co lại (erosion) loại bỏ các vật thể hoặc chi tiết nhỏ hơn structuring element.
- sau đó sẽ giãn ra lại (dilation) nhằm khôi phục hình dạng ban đầu của các vùngcòn lại.
# Ví dụ:
nếu ảnh nhị phân có các đốm trắng nhiễu ta dùng -binary_opening để loại bỏ hoàn toàn các đốm này hoặc ảnh có nhiều chi tiết nhỏ, binary_opening sẽ chỉ còn lại các vùng chính
# Code chính:
```python
data_np = np.array(data)
binary_img = data_np > 128
s = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]])
b = nd.binary_opening(binary_img, structure=s, iterations=25)
c = Image.fromarray((b * 255).astype(np.uint8))
c.show()
```
## 2.3.3 sử dụng binary_erosion
# Mục đích:
- Dùng để co đối tượng bằng cách loại bỏ pixels ở biên đối tượng, loại bỏ chi tiết nhỏ nhô ra, tách rời các đối tượng dính nhau
- Thu nhỏ vùng trắng trong ảnh nhị phân
# Nguyên lý:
- Duyệt qua ảnh với structuring element (SE)
- một điểm trắng chỉ được giữ lại nếu tất cả trong SE ứng với vùng của ảnh cũng là trắng, nếu không thoả mãn thì điểm đó sẽ bị xoá
- lặp lại nhiều vòng ( iterations0 ) sẽ làm co lại dần vùng trắng.
# Ví dụ:
- Ảnh nhị phân có các hình trắng sau nhiều vòng erosion, các hình sẽ mỏng dần hoặc biến mất hoàn toàn nếu nhỏ và các chi tiết đường răng cưu đều được làm gọn
Code chính:
```python
b = np.array(a)
s = b > 128
c = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]])
d = nd.binary_erosion(s, structure=c, iterations=50)
e = Image.fromarray((d * 255).astype(np.uint8))
```
## 2.3.4 sử dụng binary_closing
# Mục đích:
-Lấp đầy các lỗ nhỏ màu đen bên trong vùng trắng của ảnh nhị phân, làm đầy khe hở nhỏ trong đối tượng, loại bỏ nhiễu đen nhỏ bên trong vùng trắng,
- làm mượt đường biên, kết nối các thành phần gần nhau, kết nối các chi tiết bị ngắt rời nhẹ
# Nguyên Lý 
- là phép đóng ảnh nhị phân gồm hai bước là dilation( giãn) và erosion(co): thu hẹp lại hình dạng gốc. Bước dilation lấp đầy các lỗ đen sau đó erosion khôi phục lại đường viền
Ví dụ:


Code chính:
```python
img = np.array(data)
binary = img > 128
s = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
b = nd.binary_closing(binary, structure=s, iterations=50)
c = Image.fromarray((b * 255).astype(np.uint8))
c.show()
plt.imshow(c, cmap='gray')
```
Bài tập
## 1. Viết chương trình chọn LangBiang trong ảnh Đà Lạt từ thư mục exercise. Tịnh tiến vùng chọn sang phải 100px . Sử dụng phương pháp Otsu để phân vùng LangBiang theo ngưỡng 0.3, lưu vào máy với tên lang_biang.jpg và hiển thị trên màn hình
# Mục đích:
- Tách ảnh LangBiang khỏi ảnh dalat.png để thực hiện xử lý riêng
- tịnh tiến vùng chọn sang phải 100px
-Sử dụng phương pháp Otsu để phân vùng đối tượng Lang_biang bằng ngưỡng cường độ 0.3 để tạo ảnh nhị phân
- lưu ảnh kết quả quan_truong_lam_vien.jpg
# Nguyên lý:
- Tịnh tiến: Dùng hàm scipy.ndimage.shift để duy chuyển vùng ảnh theo toạ độ yêu cầu
- ảnh gốc thường sẽ có giá trị là 0-255, nên cần chia cho 255 để chuyển về [0, 1]
- áp dụng ngưỡng thủ công là 0.3 để phân biệt vùng sáng (đối tượng) và vùng tối ( nền)

# giải thích: 
Sau khi dịch phải 100px, ảnh được chuẩn hóa và áp dụng ngưỡng 0.3:
Vùng có giá trị pixel > 0.3 là trắng
Ngược lại là đen
Ví dụ:
Code chính:
```python
lang_biang_shifted = nd.shift(bmg, shift=(0, 100))

if lang_biang_shifted.max() > 1.0:
    lang_biang_norm = lang_biang_shifted / 255.0
else:
    lang_biang_norm = lang_biang_shifted
otsu_thresh = threshold_otsu(lang_biang_norm)
segmented = lang_biang_norm > otsu_thresh  
segmented_image = Image.fromarray((segmented * 255).astype(np.uint8))
segmented_image.save('lang_biang.jpg')
```
## 2. Viết chương trình chọn Hồ Xuân Hương trong ảnh Đà Lạt từ thư mục exercise. Xoay đối tượng vừa chọn 1 góc 45 độ và dùng phương pháp adaptive threshoding với ngưỡng 60 và lưu vào máy với tên là ho_xuan-huong.jpg
# Mục đích:
- Chọn vùng Hồ Xuân Hương trong ảnh dalat.jpg
- xoay ảnh góc 45 độ theo chiều kim đồng hồ
- áp dụng adaptive threshoding với offset 60
lưu ảnh kết quả 
# nguyên lý: 
- dùng hàm scipy.ndimage.rotate() để xoay ma trận ảnh theo một góc 45 độ theo chiều kim đồng hồ, tham số reshape= True giúp mở rộng khung ảnh để chứa toàn bộ vùng xoay
- threshold_local tính ngưỡng cục bộ cho từng vùng nhỏ (block) với block_size = 39 là vùng cửa sổ trượt để tính ngưỡng
offset = 60: ngưỡng được giảm bớt 60 đơn vị so với trung bình cục bộ nên ảnh tối sẽ sáng hơn, điểm ảnh nào lớn hơn ngưỡng cục bộ 60 sẽ là trắng 1, ngược lại là đen 0
Code chính:
```python
ho_xuan_huong_rotate = nd.rotate(ho_xuan_huong, 45, reshape=True)
a = np.asarray(ho_xuan_huong_rotate)
b = threshold_local(a, block_size=39, offset=60)
binary = a > b
binary_image = Image.fromarray((binary * 255).astype(np.uint8))
binary_image.save('ho_xuan_huong.jpg')
```
## 3.Viết chương trình chọn quảng trường Lâm Viên trong ảnh Đà Lạt từ thư mục exercise. Dùng phương pháp coordinate Mapping và Binary Closing cho vùng vừa chọn. Lưu vào máy với tên là quan_truong_lam_vien.jpg.
# Mục đích:
-Chọn vùng ảnh chứa quảng trường Lâm Viên từ dalat.jpg
-Áp dụng biến  bằng coordinate -mapping để làm biến dạng nhẹ ảnh.
Dùng phân ngưỡng thích nghi + binary closing để phân tách vùng và làm mịn đối tượng.
# Nguyên lý:
- Coordinate Mapping làm Biến đổi vị trí điểm ảnh một cách ngẫu nhiên để tạo hiệu ứng nhiễu .
-Threshold LocalmTính ngưỡng cục bộ cho từng vùng nhỏ, giúp phân biệt vùng sáng tối rõ ràng.
- Binary Closing Làm mịn vùng trắng bằng cách lấp đầy các lỗ nhỏ và kết nối các điểm gần nhau.
# hoạt động diễn ra:
-Sau khi biến dạng ngẫu nhiên (coordinate mapping), vùng ảnh bị nhòe nhẹ.
Sau đó dùng threshold và binary_closing để làm sạch ảnh nhị phân.
# Code chính:
```python
V, H = bmg.shape
M = np.indices((V, H))
d = 15
q = 2 * d * np.random.ranf(M.shape) - d
mp = (M + q).astype(int)
mp[0] = np.clip(mp[0], 0, V - 1)
mp[1] = np.clip(mp[1], 0, H - 1)
bmg = nd.map_coordinates(bmg, mp)
T = threshold_local(bmg, 35, offset=10)
binary = bmg > T
s = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
b = nd.binary_closing(binary, structure=s, iterations=1)
c = Image.fromarray((b * 255).astype(np.uint8))
c.save('quan_truong_lam_vien.jpg')
```
## 4. tạo menu như trong hình mẫu
## viết chương trình cho phép người dùng nhập chức năng muốn xử lý.(Có thể chọn 1 chức năng duy nhất kết hợp 2 chức năng của geometric_transformation và segment)

# Mục đích:
- Tạo chương trình cho phép người dùng chọn một hoặc hai chức năng từ hai nhóm:
- Nhóm geometric_transformation: các phép biến đổi hình học ảnh:  xoay, tịnh tiến, phóng to, biến dạng.
- Nhóm segment: các kỹ thuật phân đoạn ảnh : Otsu, adaptive thresholding, nhị phân, giãn, co ảnh.
-  xử lý ảnh linh hoạt theo yêu cầu người dùng và hiển thị kết quả.

# Nguyên lý: 
- Coordinate Mapping: tạo biến dạng ảnh.
- Rotate: xoay ảnh 
- Scale: Phóng to hoặc thu nhỏ bằng hệ số fx, fy trên trục X và Y.
- Shift: Dịch ảnh 
- Adaptive Thresholding: xử lý ảnh không đồng đều độ sáng.
- Otsu: Tìm ngưỡng tối ưu  
- Binary Dilation/Erosion: mở rộng hoặc làm co vùng trắng trong ảnh nhị phân.
# Ví dụ hoạt động:
sau khi chạy, chương trình sẽ hiện lên phần nhập lần lượt là geometric_transformation và segment, người dùng nhập các chức năng cần chọn hoặc nhấp enter để bỏ qua, sau đó chương trình sẽ sử lí ảnh và hiển thị 
người dùng nhập
geometric_transformation: Rotate
segment: Otsu
Kết quả: ảnh được xoay 45 độ và phân ngưỡng nhị phân bằng phương pháp Otsu.
# Code chính:
```python
def coordinate_mapping(image):
    V, H = image.shape
    M = np.indices((V, H))
    d = 5
    q = 2 * d * np.random.ranf(M.shape) - d
    mp = (M + q).astype(int)
    mp[0] = np.clip(mp[0], 0, V - 1)
    mp[1] = np.clip(mp[1], 0, H - 1)
    return map_coordinates(image, mp)

def rotate(image, angle=45):
    center = tuple(np.array(image.shape[::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)

def scale(image, fx=1.5, fy=1.5):
    return cv2.resize(image, None, fx=fx, fy=fy)

def shift(image, dx=30, dy=30):
    return nd_shift(image, shift=(dy, dx))

def adaptive_thresholding(image):
    T = threshold_local(image, 41, offset=10)
    return image > T

def binary_dilation(image):
    return binary_dilation(image)

def binary_erosion(image):
    return binary_erosion(image)

def otsu_thresholding(image):
    T = threshold_otsu(image)
    return image > T

print("Menu:")
print("geometric_transformation")
print("  └── coordinate_mapping")
print("  └── Rotate")
print("  └── Scale")
print("  └── Shift")
print("segment")
print("  └── Adaptive_thresholding")
print("  └── Binary_dilation")
print("  └── Binary_erosion")
print("  └── Otsu")

geo_choice = input("Nhập chức năng geometric_transformation hoặc enter bỏ qua : ").strip()
seg_choice = input("Nhập chức năng segment hoặc enter bỏ qua: ").strip()

image = cropped.copy()

geo_map = {
    "coordinate_mapping": coordinate_mapping,
    "Rotate": rotate,
    "Scale": scale,
    "Shift": shift
}
seg_map = {
    "Adaptive_thresholding": adaptive_thresholding,
    "Binary_dilation": binary_dilation_func,
    "Binary_erosion": binary_erosion_func,
    "Otsu": otsu_thresholding
}

if geo_choice in geo_map:
    image = geo_map[geo_choice](image)

if seg_choice in seg_map:
    result = seg_map[seg_choice](image)
    image = (result * 255).astype(np.uint8)
```
# Tài liệu tham khảo 
Digital Image Processing - Rafael C. Gonzalez
https://thigiacmaytinh.com/ly-thuyet-ve-phan-nguong-anh-threshold/
Slide bài giảng Nhập môn Xử lý ảnh số - Văn Lang University
