import cv2
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compare(img1, img2):
    mse = compare_mse(img1, img2)
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(to_gray(img1), to_gray(img2))
    print("mse: {}".format(mse))
    print("psnr: {}".format(psnr))
    print("ssim: {}".format(ssim))

origin_img_path = "/data_1/data/super-resolution/srgan/val/000001.jpg"
origin = cv2.imread(origin_img_path)
origin_shape = origin.shape
print(origin_shape)

target1 = cv2.blur(origin, ksize=(5,5))
print("compare to blur 5 ...")
compare(origin,target1)

target2 = cv2.blur(origin, ksize=(2,2))
print("compare to blur 2 ...")
compare(origin,target2)

target3 = cv2.blur(origin, ksize=(10,10))
print("compare to blur 10 ...")
compare(origin,target3)



cv2.imshow("origin", origin)
cv2.imshow("target1", target1)
cv2.imshow("target2", target2)
cv2.waitKey(0)