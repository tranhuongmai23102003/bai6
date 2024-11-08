# Import các thư viện cần thiết
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# Đọc ảnh vệ tinh
image = cv2.imread('vetinh6.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang không gian màu RGB

# Chuyển ảnh thành mảng 2D, mỗi điểm ảnh là một vector 3 chiều đại diện cho màu
pixel_values = image.reshape((-1, 3)).astype(np.float32)

# 1. Phân cụm bằng K-means
k = 3  # Số cụm
kmeans = KMeans(n_clusters=k, random_state=0)
labels_kmeans = kmeans.fit_predict(pixel_values)
segmented_image_kmeans = labels_kmeans.reshape(image.shape[:2])  # Đổi nhãn về ảnh 2D

# 2. Phân cụm bằng Fuzzy C-means (FCM)
n_clusters = 3  # Số cụm
fcm_data = pixel_values.T  # Chuyển vị dữ liệu cho FCM
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    fcm_data, n_clusters, m=2, error=0.005, maxiter=1000, init=None
)
labels_fcm = np.argmax(u, axis=0)  # Nhãn dựa vào mức độ thành viên cao nhất
segmented_image_fcm = labels_fcm.reshape(image.shape[:2])

# Hiển thị ảnh gốc và kết quả phân cụm
plt.figure(figsize=(15, 5))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Kết quả K-means (màu xám)
plt.subplot(1, 3, 2)
plt.imshow(segmented_image_kmeans, cmap='gray')  # Bảng màu xám
plt.title("K-means Clustering")
plt.axis('off')

# Kết quả Fuzzy C-means (màu xám)
plt.subplot(1, 3, 3)
plt.imshow(segmented_image_fcm, cmap='gray')  # Bảng màu xám
plt.title("Fuzzy C-means Clustering")
plt.axis('off')

plt.show()
