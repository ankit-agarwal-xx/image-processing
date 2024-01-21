import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

image = mping.imread("images.jpeg") #replace with your image
if image.shape[-1] == 3:
    image = np.dot(image[...,:3], [0.28, 0.59, 0.11])

hist, bins = np.histogram(image, bins=256, range=(0, 256))
hist_prob = hist / float(image.size)
cumsum_prob = np.cumsum(hist_prob)
mean_intensity = np.sum(np.arange(256) * hist_prob)

best_threshold = 0
max_variance = 0
for threshold in range(1, 256):
    probl = cumsum_prob[threshold]
    prob2 = 1 - probl
    mean1 = np.sum(np.arange(threshold) * hist_prob[:threshold])
    mean2 = (mean_intensity - (probl * mean1)) / prob2
    between_class_variance = probl * prob2 * (mean1 - mean2) ** 2
    if between_class_variance > max_variance:
        max_variance = between_class_variance
        best_threshold = threshold

otsu_threshold = (image >= best_threshold) * 255

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('original image')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.title("Otsu's thresholding")
plt.imshow(otsu_threshold, cmap='gray')
plt.tight_layout()
plt.show()
