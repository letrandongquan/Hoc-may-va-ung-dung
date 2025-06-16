
## LAB3: Biến đổi hình học

## Viết chương trình biến đổi ảnh
## 1.1 chọn đối tượng trong ảnh
phép trích ảnh nhỏ trong một ảnh lớn ban đầu. chương trình yêu cầu chọn một vật thể trong ảnh gốc ra.
## Công Nghệ sử dụng
output = input_array[start_dim1:end_dim1, start_dim2:end_dim2]
đây là câu lệnh dùng để cắt một phần của mảng dữa liệu với
start_dim1:end_dim1 là trục y và start_dim2:end_dim2 là trục x
vd:
bmg = data[800:1200, 570:980]
### giải thích hoạt động
```python
# xử dụng chương trình hiển thị ra ảnh cùng với trục nếu chưa biết kích thước ảnh cần cần cắt
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('fruit.jpg')
bmg = data[800:1200, 570:980]
print(data.shape)

iio.imsave('orange.jpg', bmg)
plt.imshow(bmg)
plt.show()

# chương trình chính
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('fruit.jpg')
# sau khi ta xác định được vật nằm ở vị trí nào trong ảnh thông qua trục, nhập số liệu vào chương trình để tiến hành cắt 
bmg = data[800:1200, 570:980]
print(data.shape)
# chương trình cắt phần ta chọn, lưu và hiển thị
iio.imsave('orange.jpg', bmg)
plt.imshow(bmg)
plt.show()
```
## 1.2 Tịnh tiến đơn
phép tịnh tiến là phép biến đổi ảnh theo hướng ngang của trục x hoặc dọc của trục y mà không làm thay đổi kích thước ảnh, bài tạp65 yêu cầu thực hiện dịch chuyển ảnh theo chiều dọc và ngang
## Công Nghệ sử dụng
hàm scipy.ndimage.shift() được xử dụng để tịnh tuyến ảnh hoặc dữ liệu
vd: 
bdata = nd.shift(data, (100, 25)), trong đó 100 thuộc trục y và 25 là trục x
## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('fruit.jpg', mode='F')
# khi ta muốn ảnh dịch chuyển sang bên nào, ta nhập số liệu phù hợp vào ,chương trình sau khi chạy sẽ hiển thị ảnh dịch chuyển theo yêu cầu
bdata = nd.shift(data, (100, 25))

plt.imshow(bdata, cmap='gray')
plt.show()
```
## 1.3 Thay đổi kích thước ảnh
làm tăng hoặc giảm đi kích thước của ảnh
## công Nghệ sử dụng
output_array = scipy.ndimage.zoom(input_array, zoom_factor), lệnh này cho phép phóng to ảnh, thu nhỏ ảnh, tạo hiệu ứng thu phóng...
vd: bdata = nd.zoom(data, 2) cho phép phóng to ảnh lên gấp 2 lần

## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('fruit.jpg')
print(data.shape)
# phóng to ảnh lên 2 lần theo tất cả các chiều
bdata = nd.zoom(data, 2)
print(bdata.shape)
# phóng to ảnh có kiểm soát, chiều cao y sẽ zoom lên 2x và chiều rộng x sẽ zoom lên 2x, chiều kệnh màu giữ nguyên
data2 = nd.zoom (data, (2, 2, 1))
print(data2.shape)
# thu nhỏ ảnh, chiều cao y sẽ thu nhỏ đi 0.5x, chiều rộng x thu nhỏ đi 0.9x kênh màu giữ nguyên
data3 = nd.zoom(data, (0.5, 0.9, 1))
plt.imshow(data3)
plt.show()
```
## 1.4 xoay ảnh ( rotate)
dùng hàm rotate(image, degree) để xoay ảnh theo chiều được yêu cầu
## công Nghệ sử dụng
output_array = scipy.ndimage.rotate(image, degree)
cho phép xoay ảnh theo chỉ định
vd: d1 = nd.rotate(data, 20) với data là image và 20 là degree( góc xoay)
## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('fruit.jpg')
print(data.shape)
# lệnh này cho phép xoay ảnh 20 độ ngược theo chiều kim đồng hồ, sau đó hiển thị ảnh đã xoay, reshape=True( mặc định) sau khi xoay, ảnh sẽ không mất phần nào
d1 = nd.rotate(data, 20)
plt.imshow(d1)
plt.show()
#ta có reshape=False, các ảnh bị xoay ra ngoài sẽ bị cắt bỏ 
d2 = nd.rotate(data, 20, reshape=False)
plt.imshow(d2)
plt.show()
```
## 1.5 Dilation và Erosion
Dùng để loại bỏ những pixel nhiễu
Dilation thay thế pixel toạ độ (i, j) bằng giá trị lớn nhất của những pixel lân cận (kề)
erosion thay thế pixel toạ độ (i,j) bằng giá trị nhỏ nhất của những pixel lân cận(kề).
## công Nghệ sử dụng
output = scipy.ndimage.binary_dilation(binary), sử dụng để loại bỏ nhiễu đen trong vùng trắng, lấp đầy các lỗ đen
d2 = nd.binary_dilation(binary, iterations=3), chống nhiễu, loại bỏ các nhiễu lớn,...
vd: d1 = nd.binary_dilation(binary)
d2 = nd.binary_dilation(binary, iterations=3)
## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('world_cup.jpg', mode='F')
print(data.shape)
# loại bỏ các nhiễu đen có kích thước <= 3x3 pixel bằng phép giãn (dilation), mỗi pixel trắng (1) sẽ lan toả ra các pixel lân cận lấp đầy khoảng trống xung quanh
d1 = nd.binary_dilation(binary)
plt.imshow(d1, cmap='gray')
plt.show()
# ta dùng phép dãn nở 3 lẩn2 liên tiếp ( iterations=3 ), mỗi lần giãn sử dụng kết quả của lần giãn trước đó, làm vùng trắng mở rộng mạnh hơn
d2 = nd.binary_dilation(binary, iterations=3)
plt.imshow(d2, cmap='gray')
plt.show()
```
## 1.6 Coordinate Mapping
Cho phép tạo hàm mới do người dùng định nghĩa ngoài các hàm có sẵn như shifting, rotate,..

## công Nghệ sử dụng
np.indices() tạo mảng 2d chứa toạ độ (x, y) của từng pixel trong ảnh
np.clip(mp[], , ) giới hạn toạ độ trong biên ảnh, đảm bảo toạ độ sau khi biến dạng không vược ra ngoài phạm vi ảnh
np.random.ranf(M.shape) - d sử dụng sinh ngẫu nhiên các số thuộc [0,1) có cùng shape với <>
d1 = nd.map_coordinates(data, mp)
map_coordinates ánh xạ lại giá trị pixel từ ảnh gốc data theo toạ độ mới mp

## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('world_cup.jpg', mode='F')
print(data.shape)
# lấy kích thước của ảnh
V, H = data.shape
# sử dụng np.indices() để tạo lưới toạ độ gốc với M[0] tương ứng với chỉ số cột y và M[1] tương ứng với x
M = np.indices( (V,H))
d = 5
q = 2 * d * np.random.ranf(M.shape) - d # q gây nhiễu toạ độ và gây lệch khỏi vị trí mẫu
mp = (M + q).astype(int)
#thực hiện lấy mẫu ảnh theo toạ độ mp, do mp lệch nên ảnh của kết quả sẽ bị biến dạng ngẫu nhiên
nd.map_coordinate() 
d1 = nd.map_coordinates(data, mp)
plt.imshow(d1, cmap='gray')
plt.show()
```

## 1.7 Biển đổi chung (Generic Tranformation)
ta sử dụng generic transformation khi muốn biến đổi các ảnh chung phép toán do người dùng định nghĩa

## công Nghệ sử dụng
GeoFun(outcoord) là hàm biến đổi toạ độ, nhận vào toạ độ điểm đích (y, x) và tính ra toạ độ nguồn (a, b) từ đó ra giá trị pixel
vd trong bài 
a = 10 * np.cos(outcoord[0]/10.0) + outcoord[0]
b = 10 * np.cos(outcoord[1]/10.0) + outcoord[1]
với outcoord[0] là toạ độ dòng (y)
outcoord[1] là toạ độ dòng (x)
nd.geometric_transform(data,GeoFun) : biến đổi hình học theo hàm geofun

## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
# ta sử dụng hàm GeoFun(outcoord) nhận vào toạ độ điểm đích (y, x) và tính ra toạ độ nguồn (a, b) từ đó ra giá trị pixel
def GeoFun(outcoord):
    a = 10 * np.cos(outcoord[0]/10.0) + outcoord[0]
    b = 10 * np.cos(outcoord[1]/10.0) + outcoord[1]
    return a, b
data = iio.imread('world_cup.jpg', mode='F')
# sử dụng nd.geometric_transform biến đổi hình học ảnh đầu vào tông qua GeoFun là hàm ánh xạ ngược
d1 = nd.geometric_transform(data,GeoFun)
plt.imshow(d1, cmap='gray')
plt.show()
```

## Bài Tập 

## 1. Viết chương trình chọn quả kiwi từ ảnh colorful-ripe-tropical-fruits.jpg trong thư mục exercise. tịnh tuyến quả kiwi sang 30 pixels
bài này sử dụng kết hợp phép trích ảnh và tịnh tuyến để thực hiện cắt đối tượng trong ảnh gốc ra và tịnh tuyết ảnh mới sang 30 pixel
## công Nghệ sử dụng
output = input_array[start_dim1:end_dim1, start_dim2:end_dim2]
đây là câu lệnh dùng để cắt một phần của mảng dữa liệu với
start_dim1:end_dim1 là trục y và start_dim2:end_dim2 là trục x

hàm scipy.ndimage.shift() được xử dụng để tịnh tuyến ảnh hoặc dữ liệu

## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('colorful-ripe-tropical-fruits.jpg', mode='F')
# lấy kích thước vị trí cần cắt ở trục của ảnh gốc, sau đó đưa vào dòng lệnh tương ứng với 920:1100 là trục y và 380:580 là trục x
bmg = data[920:1100, 380:580]
# sau đó sử dụng nd.shift(bmg, (0,30)) để dịch chuyển ảnh sang 30 pixel mà không khiến cho kích thước ảnh bị thay đổi

kiwi = nd.shift(bmg, (0, 30))  
# lưu ảnh thành kiwi.jpg ảnh xám và hiển thị
iio.imsave('kiwi.jpg', kiwi.astype(np.uint8))
kiwi_img = iio.imread('kiwi.jpg')
plt.imshow(kiwi_img, cmap='gray')
plt.show()
```

## 2.viết chương trình chọn quả đu đủ và quả dưa hấu từ  colorful-ripe-tropical-fruits.jpg trong thư mục exercise, đổi màu hai đối tượng này
bài này yêu cầu thực hiện cắt ảnh gốc hai quả đu đủ và dưa hấu ra và đổi màu cả hai ảnh
## công Nghệ sử dụng
output = input_array[start_dim1:end_dim1, start_dim2:end_dim2]
đây là câu lệnh dùng để cắt một phần của mảng dữa liệu với
start_dim1:end_dim1 là trục y và start_dim2:end_dim2 là trục x
image[:, :, [i, j]] = image[:, :, [j, i]] đổi màu từ red sang green
image[:, :, k] = np.clip(image[:, :, 0] + 100, 0, 255) tăng độ sáng blue dựa trên red
ig, axes = plt.subplots tạo hàng và cột
## giải thích cách hoạt động
```python
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt

img = iio.imread('colorful-ripe-tropical-fruits.jpg')
# sau khi xác định số liệu chỉ vị trí của hai ảnh trên trục, cắt hai ảnh ra
papaya = img[300:820, 100:750]
watermelon= img[200:1200, 1650:2500]
# đổi màu cho 2 ảnh vừa cắt ra 
papaya[:, :, [0, 1]] = papaya[:, :, [1, 0]]
watermelon[:, :, 2] = np.clip(watermelon[:, :, 0] + 100, 0, 255)
#tạo 1 hàng và 2 cột biểu đồ để tách riêng 2 ảnh ra khi hiển thị với axes[0] là ảnh papaya bên trái và axes[1] là ảnh watermelon phía bên phải, hiển thị ảnh
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(papaya)
axes[1].imshow(watermelon)
plt.tight_layout()
plt.show()
```
## 3. viết chương trình chọn ngọn núi và con thuyền từ ảnh quang_ninh.jpg trong thư mục exercise. xoay 2 đối tượng này góc 45 độ và lưu vào máy
bài tập này yêu cầu cắt 2 ảnh con thuyền và ngôn núi từ ảnh gốc quang ning.jpg và xoay cả hai góc 45 độ và lưu vào máy
## công Nghệ sử dụng
output = input_array[start_dim1:end_dim1, start_dim2:end_dim2]
đây là câu lệnh dùng để cắt một phần của mảng dữa liệu với
start_dim1:end_dim1 là trục y và start_dim2:end_dim2 là trục x
nd.rotate(input, angel, reshape=True) hàm để xoay ảnh
ig, axes = plt.subplots tạo hàng và cột
## giải thích cách hoạt động
```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
# sau khi xác định toạ độ của 2 vật con thuyền và ngọn núi, nhập số liệu để tiến hành cắt hai hình này ra từ ảnh gốc quanhning.jpg
image = iio.imread('quang_ninh.jpg')
ngon_nui = image[0:330, 410: 700]
con_thuyen = image[420:580, 500:700]
#nd.rotate(input, angel, reshape=True) nhập 45 để xoay cả hai hình theo ngược chiều kim đồng hồ một gốc 45 độ
xoay_ngon_nui = nd.rotate(ngon_nui, 45, reshape=True)
xoay_con_thuyen = nd.rotate(con_thuyen, 45, reshape=True)
iio.imsave('ngon_nui.jpg', xoay_ngon_nui)
iio.imsave('con_thuyen.jpg', xoay_con_thuyen)
# tạo 1 hàng và 2 cột biểu đồ để tách riêng 2 ảnh ra khi hiển thị với axes[0] là ảnh ngọn núi bên trái và axes[1] là ảnh con thuyền phía bên phải
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(xoay_ngon_nui)
axes[1].imshow(xoay_con_thuyen)

plt.tight_layout()
plt.show()
```

## 4. viết chương trình chọn ngôi chùa từ ảnh pagoda.jpg trong thư mục exercise. Tăng kích thước ngôi chùa lên 5 lần và lưu vào máy
bài tập cắt ngôi chùa từ ảnh gốc là pagoda.jpg trong thư mục exercise và zoom ngôi chùa lên 5 lần đồng thời sẽ lưu vào máy và lưu kết kết quả vào máy
## công Nghệ sử dụng
output = input_array[start_dim1:end_dim1, start_dim2:end_dim2]
đây là câu lệnh dùng để cắt một phần của mảng dữa liệu với
start_dim1:end_dim1 là trục y và start_dim2:end_dim2 là trục x
output_array = scipy.ndimage.zoom(input_array, zoom_factor), lệnh này cho phép phóng to ảnh, thu nhỏ ảnh, tạo hiệu ứng thu phóng...
## giải thích cách hoạt động

```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pylab as plt
# ta cắt hình ảnh ngôi chùa từ ảnh gốc pagoda.jpg sau đó sử dụng nd.zoom để tiến hành tăng kích thước với (chiều cao, chiều rộng, kênh màu), chiều cao X5, chiều rộng X5 và kênh màu giữ nguyên
image = iio.imread('pagoda.jpg')
ngoi_chua = image[130:210, 0:600]
tang_kich_thuoc_chua = nd.zoom(ngoi_chua, (5, 5, 1))
# lưu ảnh và hiển thị ảnh đã zoom
iio.imsave('pagoda_zoom_5_lan.jpg', tang_kich_thuoc_chua)
plt.imshow(tang_kich_thuoc_chua)
plt.show()
```

## 5. viết chương trình tạo menu
- tịnh tiến
- xoay
- phóng to
- thu nhỏ
- coordinate map
khi chọn phím T, X, P, H, C thì hỏi muốn thực hiện trên hình nào từ trong thư mục exercise, người dùng chọn hình nào thì chọn phép biến đổi như trên hình đó

bài yêu cầu hiển thị menu cho người dùng thấy các chức năng phép biến đổi ảnh, sau đó nhập kí tự và chọn ảnh để thực hiện phép biến đổi 


## công Nghệ sử dụng
import os 
- lệnh import os sử dụng để cung cấp chức năng tương tác với hệ thống file, tạo đường dẫn tập tin exercise trong chương trình này  vd: folder = 'exercise'
nd.shift(img, shift=(shift_y, shift_x, 0)) tịnh tuyến ảnh theo phương dọc y và phương ngang x
nd.shift(img, shift=(shift_y, shift_x, 0)) lệnh xoay ảnh quanh một tâm góc 
nd.zoom(img, (factor, factor, 1)) lệnh dùng để phóng to hoặc thu nhỏ
coordinate_map(img) làm biến dạng ảnh, ánh xạ vị trí mới mới cho ảnh đầu vào
## giải thích cách hoạt động

```python
import numpy as np
import scipy.ndimage as nd
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os

folder = 'exercise'
# hàm hiển thị kết quả sau khi người dùng thao tác với các phép biến đổi ảnh
def hien_thi_anh(img, title="Kết quả"):
    plt.imshow(img.astype(np.uint8))
    plt.show()

# các hàm chứa các chức năng xử lý ảnh 
# phép tịnh tiến 
def tinh_tien(img, shift_x=30, shift_y=30):
    return nd.shift(img, shift=(shift_y, shift_x, 0))

# phép xoay
def xoay(img, angle=45):
    return nd.shift(img, shift=(shift_y, shift_x, 0))

# phép phóng to hoặc thu nhỏ
def phong_to_hoac_thu_nho(img, factor):
    return nd.zoom(img, (factor, factor, 1))

# phép biến dạng ảnh
def coordinate_map(img):
    if img.ndim == 3:
        data = img.mean(axis=2) 
    else:
        data = img

    print(data.shape)
    V, H = data.shape
    M = np.indices((V, H))
    d = 5
    q = 2 * d * np.random.ranf(M.shape) - d
    mp = (M + q).astype(int)
    mp[0] = np.clip(mp[0], 0, V - 1)
    mp[1] = np.clip(mp[1], 0, H - 1)
    d1 = nd.map_coordinates(data, mp)
    plt.imshow(d1, cmap='gray')
    plt.show()


# Lấy danh sách file ảnh trong thư mục exercise, nếu không tìm thấy thì in ra dòng lệnh báo và dừng
files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg'))]
if not files:
    print("Không tìm thấy ảnh trong thư mục.")
    exit()

# Hiển thị danh sách ảnh
print("Danh sách hình ảnh trong thư mục 'exercise':")
for i, f in enumerate(files):
    print(f"{i+1}. {f}")

# Chọn ảnh, người dùng sẽ nhập số thứ tự ảnh, nếu nhập số nhỏ hơn 1 hoặc lớn hơn số ảnh hiện có trong file thì sẽ báo lỗi và thoát
index = int(input("Chọn số thứ tự ảnh muốn xử lý: ")) - 1
if index < 0 or index >= len(files):
    print("Lựa chọn không hợp lệ.")
    exit()

file_path = os.path.join(folder, files[index])
img = iio.imread(file_path)

# phần hiển thị lựa chọn cho người dùng cùng 5 lựa chọn, người dùng nhập kí tự in hoa tương ứng với lựa chọn cần xử lí ảnh
print("\nChọn thao tác:")
print("T - Tịnh tiến")
print("X - Xoay ảnh")
print("P - Phóng to")
print("H - Thu nhỏ")
print("C - Coordinate Map")

choice = input("Nhập lựa chọn của bạn (T/X/P/H/C): ").strip().upper()

# Các bước xử lý sau khi người dùng chọn các lựa chọn, tuỳ thuộc vào lựa chọn của người dùng mà chương trình sẽ xử lí yêu cầu tương ứng:
# với tịnh tiến là nhập các trị cần dịch chuyển ảnh
if choice == 'T':
    dx = int(input("Nhập giá trị tịnh tiến ngang (dx): "))
    dy = int(input("Nhập giá trị tịnh tiến dọc (dy): "))
    result = tinh_tien(img, dx, dy)
    hien_thi_anh(result, "Ảnh sau khi tịnh tiến")
# với rotate người dùng cần nhập degree cần xoay, ảnh sẽ xoay theo yêu cầu theo ngược chiều kim đồng hồ
elif choice == 'X':
    angle = float(input("Nhập góc xoay (độ): "))
    result = xoay(img, angle)
    hien_thi_anh(result, "Ảnh sau khi xoay")
# chọn P sẽ phòng to ảnh, người dùng nhập số lớn hơn hơn 1 và ảnh sẽ được zoom lên, ngược lai thu nhỏ H, người dùng sẽ nhập số nhỏ  hơn 1
elif choice == 'P':
    factor = float(input("Nhập hệ số phóng to (>1): "))
    result = phong_to_hoac_thu_nho(img, factor)
    hien_thi_anh(result, "Ảnh sau khi phóng to")

elif choice == 'H':
    factor = float(input("Nhập hệ số thu nhỏ (<1): "))
    result = phong_to_hoac_thu_nho(img, factor)
    hien_thi_anh(result, "Ảnh sau khi thu nhỏ")
# khi người dùng chọn C tương ứng với coordinate_map thì chương trình sẽ đi vào gọi hàm coordinate_map và truyền ảnh mà người dùng đã chọn sau đó tạo hiệu ứng biến dạng

elif choice == 'C':
    coordinate_map(img)

else:
    print("Lựa chọn không hợp lệ.")
    ```
