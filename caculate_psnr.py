from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

origin = np.array(Image.open(r'C:\Users\Administrator\Desktop\新建文件夹\img_1_gt.png').convert('RGB'))
# img3 = np.array(Image.open(r'C:\Users\Administrator\Desktop\新建文件夹\image sr3.png').convert('RGB'))
img4 = np.array(Image.open(r'C:\Users\Administrator\Desktop\新建文件夹\image sr.png').convert('RGB'))
# img_bicubic = np.array(Image.open(r'C:\Users\Administrator\Desktop\新建文件夹\bicubic.png').convert('RGB'))
# print(psnr(origin, img3))
print(psnr(origin, img4))
# print(psnr(origin, img_bicubic))
# print(ssim(origin, img3, multichannels=True,channel_axis=2))
print(ssim(origin, img4, multichannels=True,channel_axis=2))
# print(ssim(origin, img_bicubic, multichannels=True,channel_axis=2))
