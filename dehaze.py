import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from exposure_enhancement import enhance_image_exposure


def PSNR(res_img, original_img):
    mse = 0.0
    for i in range(928):
        for j in range(1400):
            for k in range(3):
                mse += (res_img[i][j][k] - original_img[i][j][k]) ** 2
    new_mse = mse / (928 * 1400 * 3)
    max_original_2 = np.max(original_img) ** 2
    psnr = 10 * np.log10(max_original_2 / new_mse)
    return psnr


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark


def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = np.argsort(darkvec,axis=0);
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind]/A[0, ind]

    transmission = 1 - omega*DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q


def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t, tx)

    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind])/t + A[0, ind]

    return res


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = './image/029.jpg'

    src = cv2.imread(fn)

    enhanced_image = enhance_image_exposure(src, gamma=0.4, lambda_=0.15, dual=True,
                                            sigma=3, bc=1, bs=1, be=1, eps=1e-3)

    I = enhanced_image.astype('float64') / 255
 
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    # cv2.imshow("dark",dark)
    # cv2.imshow("t",t)
    # cv2.imshow('I',src)
    # cv2.imshow('J',J)
    cv2.imwrite("./image/029_dcp_enhanced.png", J*255)

    I = src.astype('float64') / 255

    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    cv2.imwrite("./image/029_dcp_no_enhance.png", J * 255)

    img_1 = cv2.imread(fn)
    img_2 = enhanced_image
    img_3 = cv2.imread('./image/029_dcp_no_enhance.png')
    img_4 = cv2.imread('./image/029_dcp_enhanced.png')
    img_5 = cv2.imread('./image/029_no_enhance_cnn.png')
    img_6 = cv2.imread('./image/029_enhance_cnn.png')

    img_lst = [img_1, img_3, img_5, img_2, img_4, img_6]
    img_name = ["original", "original_dcp", "original_cnn",
                "enhanced", "enhanced_dcp", "enhanced_cnn"]
    fig = plt.figure(figsize=(9, 5))
    columns = 3
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(img_name[i-1])
        plt.imshow(img_lst[i-1])
    plt.show()

    cv2.waitKey(0)