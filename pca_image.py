import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('Lenna.png')
print('original image matrix', img)
# plt.imshow(img)
# plt.show(block=True)
# print('Size of image: ', img.shape)

img_r = np.zeros((img.shape[0], img.shape[1]))
img_g = np.zeros((img.shape[0], img.shape[1]))
img_b = np.zeros((img.shape[0], img.shape[1]))

thresh = 0.01 # threshold for singular values
for i in range(3):
    img_cp = img[:,:,i]
    u,s,v = np.linalg.svd(img_cp, full_matrices = True)
    s_norm = s/np.linalg.norm(s)
    # s_norm = s;
    for j in range(220):
        if abs(s_norm[j]) > thresh and i == 0:
            img_r += s[j]*np.outer(u[:,j], v[j,:])
        elif abs(s_norm[j]) > thresh and i == 1:
            img_g += s[j]*np.outer(u[:,j], v[j,:])
        elif abs(s_norm[j]) > thresh and i == 2:
            img_b += s[j]*np.outer(u[:,j], v[j,:])

img_comp = np.stack( (img_r, img_g, img_b), axis = 2)
print('compressed image matrix', img_comp)

plt.imshow(img_comp)
plt.show(block=True)
