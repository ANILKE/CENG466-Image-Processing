import math
import os


import cv2
import numpy as np
from scipy.linalg import hadamard
from scipy import signal, ndimage
from PIL import Image
INPUT_PATH = "./THE_2 images/"
OUTPUT_PATH = "./Outputs/"




def image_read(img_path):
    image = cv2.imread(img_path)
    return image

def image_write(img, output_path):
    cv2.imwrite(output_path, img)

def fourier_transform(img):  #RGB ye ayurmayı unutmuştuk

    b, g, r = cv2.split(img)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[],[],[]]
    for i in range(3):
        f = np.fft.fft2(r_g_b_dic[i])
        fshift = np.fft.fftshift(f)
        result[i] = 20 * np.log(np.abs(fshift))

    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge



def hadamard_transform(image):  #Matrix mult yanlış yapmıştık önce sağdan çarptık sadece
        org_dim=(image.shape[1],image.shape[0])
        N = math.ceil(math.log2(max(image.shape[0],image.shape[1])))
        N = int(math.pow(2, N))
        max_size = N
        dim = (max_size,max_size)
        res = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        b, g, r = cv2.split(res)
        r_g_b_dic = {0: b, 1: g, 2: r}
        result = [[], [], []]
        hadamard_m = hadamard(max_size)
        for i in range(3):
            result[i] =np.multiply(r_g_b_dic[i], hadamard_m)
        image_merge = cv2.merge([result[2], result[1], result[0]])
        res = cv2.resize(image_merge.astype(float), org_dim, interpolation=cv2.INTER_AREA)
        return res

def cosine_transform(image):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        imf = np.float32(r_g_b_dic[i])
        dst = cv2.dct(imf, cv2.DCT_INVERSE)

        result[i] = dst
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge

def ideal_low_pass(image, freq):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        filter_size = freq

        freq_image_centered = np.fft.fftshift(freq_image)
        low_pass = np.fft.fft2(np.zeros((freq_image_centered.shape[0], freq_image_centered.shape[1])))
        low_pass[
        int(freq_image_centered.shape[0] / 2) - filter_size - 1:int(freq_image_centered.shape[0] / 2) + filter_size,
        int(freq_image_centered.shape[1] / 2) - filter_size - 1: int(freq_image_centered.shape[1] / 2) + filter_size] = freq_image_centered[
                                                               int(freq_image_centered.shape[0] / 2) - filter_size - 1:int(
                                                                   freq_image_centered.shape[0] / 2) + filter_size,
                                                               int(freq_image_centered.shape[1] / 2) - filter_size - 1: int(
                                                                   freq_image_centered.shape[1] / 2) + filter_size]
        result[i] = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge

def ideal_high_pass(image, freq):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        filter_size = freq

        freq_image_centered = np.fft.fftshift(freq_image)
        high_pass = np.copy(freq_image_centered)
        high_pass[
        int(freq_image_centered.shape[0] / 2) - filter_size - 1:int(freq_image_centered.shape[0] / 2) + filter_size,
        int(freq_image_centered.shape[1] / 2) - filter_size - 1: int(freq_image_centered.shape[1] / 2) + filter_size] = 0
        result[i] = np.fft.ifft2(np.fft.ifftshift(high_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge

def gaussian_low_pass(image, freq):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        filter_size = freq

        freq_image_centered = np.fft.fftshift(freq_image)
        low_pass = np.fft.fft2(np.zeros((freq_image_centered.shape[0], freq_image_centered.shape[1])))
        for x in range(freq_image_centered.shape[0]):
            for y in range(freq_image_centered.shape[1]):
                low_pass[x, y] = freq_image_centered[x, y] * np.exp(
                    -((x - int(freq_image_centered.shape[0] / 2)) ** 2 + (y - int(freq_image_centered.shape[1] / 2)) ** 2) / (
                                2 * filter_size ** 2))
        result[i] = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge

def gaussian_high_pass(image, freq):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        filter_size = freq

        freq_image_centered = np.fft.fftshift(freq_image)
        high_pass = np.copy(freq_image_centered)
        for x in range(freq_image_centered.shape[0]):
            for y in range(freq_image_centered.shape[1]):
                high_pass[x, y] = freq_image_centered[x, y] * (1 - np.exp(
                    -((x - int(freq_image_centered.shape[0] / 2)) ** 2 + (y - int(freq_image_centered.shape[1] / 2)) ** 2) / (
                                2 * filter_size ** 2)))
        result[i] = np.fft.ifft2(np.fft.ifftshift(high_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge
#
#
# def butterworth_low_pass(image, freq, n):
#     b, g, r = cv2.split(image)
#     r_g_b_dic = {0: b, 1: g, 2: r}
#     result = [[], [], []]
#     for i in range(3):
#         b, g, r = cv2.split(image)
#         r_g_b_dic = {0: b, 1: g, 2: r}
#         result = [[], [], []]
#         for i in range(3):
#             freq_image = np.fft.fft2(r_g_b_dic[i])
#             filter_size = freq
#             n = 1
#             M, N = freq_image.shape
#             H = np.zeros((M, N), dtype=np.float64)
#             for x in range(M):
#                 for y in range(N):
#                     D = np.sqrt(((x - M / 2) ** 2 + (y - N / 2) ** 2))
#                     H[x, y] =  (1 / (1 + D / filter_size) ** 2)
#
#             low_pass = np.multiply(freq_image, H)
#             result[i] = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
#         image_merge = cv2.merge([result[2], result[1], result[0]])
#         return image_merge
#
# def butterworth_high_pass(image, freq, n):
#     b, g, r = cv2.split(image)
#     r_g_b_dic = {0: b, 1: g, 2: r}
#     result = [[], [], []]
#     for i in range(3):
#         freq_image = np.fft.fft2(r_g_b_dic[i])
#         filter_size = freq
#         n = 1
#         M, N = freq_image.shape
#         H = np.zeros((M, N), dtype=np.float64)
#         for x in range(M):
#             for y in range(N):
#                 D = np.sqrt(((x - M / 2) ** 2 + (y - N / 2) ** 2))
#                 H[x, y] =1-( 1 / (1 + D/filter_size) ** 2)
#
#         low_pass = np.multiply(freq_image, H)
#         result[i] = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
#     image_merge = cv2.merge([result[2], result[1], result[0]])
#     return image_merge
#
#
def band_pass_filter(image, freq, filter_size,filter2_size):

    nyquist = 0.5 * freq
    low = filter_size / nyquist
    high = filter2_size / nyquist
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        freq_image_centered = np.fft.fftshift(freq_image)
        band_pass = np.copy(freq_image_centered)
        for x in range(freq_image_centered.shape[0]):
            for y in range(freq_image_centered.shape[1]):
                if low <= np.sqrt((x - int(freq_image_centered.shape[0] / 2)) ** 2 + (
                        y - int(freq_image_centered.shape[1] / 2)) ** 2) <= high:
                    band_pass[x, y] = freq_image_centered[x, y]
                else:
                    band_pass[x, y] = 0
        result[i] = np.fft.ifft2(np.fft.ifftshift(band_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge
def band_reject_filter(image, freq, filter_size,filter2_size):
    nyquist = 0.5 * freq
    low = filter_size / nyquist
    high = filter2_size / nyquist
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        freq_image_centered = np.fft.fftshift(freq_image)
        band_pass = np.copy(freq_image_centered)
        for x in range(freq_image_centered.shape[0]):
            for y in range(freq_image_centered.shape[1]):
                if low <= np.sqrt((x - int(freq_image_centered.shape[0] / 2)) ** 2 + (
                        y - int(freq_image_centered.shape[1] / 2)) ** 2) <= high:
                    band_pass[x, y] = freq_image_centered[x, y]
                else:
                    band_pass[x, y] = 0
        band_reject = 1 - band_pass
        result[i] = np.fft.ifft2(np.fft.ifftshift(band_reject)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge

def butterworth_low_pass(image, freq, order):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        filter_size = freq

        freq_image_centered = np.fft.fftshift(freq_image)
        low_pass = np.fft.fft2(np.zeros((freq_image_centered.shape[0], freq_image_centered.shape[1])))
        for x in range(freq_image_centered.shape[0]):
            for y in range(freq_image_centered.shape[1]):
                low_pass[x, y] = freq_image_centered[x, y] * 1 / (1 + ((x - int(freq_image_centered.shape[0] / 2)) ** 2 + (
                            y - int(freq_image_centered.shape[1] / 2)) ** 2) / (filter_size ** 2)) ** order
        result[i] = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge

def butterworth_high_pass(image, freq, order):
    b, g, r = cv2.split(image)
    r_g_b_dic = {0: b, 1: g, 2: r}
    result = [[], [], []]
    for i in range(3):
        freq_image = np.fft.fft2(r_g_b_dic[i])
        filter_size = freq

        freq_image_centered = np.fft.fftshift(freq_image)
        high_pass = np.copy(freq_image_centered)
        for x in range(freq_image_centered.shape[0]):
            for y in range(freq_image_centered.shape[1]):
                high_pass[x, y] = freq_image_centered[x, y] * (1 / (1 + ((x - int(freq_image_centered.shape[0] / 2)) ** 2 + (
                            y - int(freq_image_centered.shape[1] / 2)) ** 2) / (filter_size ** 2)) ** order)
        result[i] = np.fft.ifft2(np.fft.ifftshift(high_pass)).real
    image_merge = cv2.merge([result[2], result[1], result[0]])
    return image_merge


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    #step 2
        #image 1
    img = image_read(INPUT_PATH + "1.png")
    output = fourier_transform(img)
    image_write(output, OUTPUT_PATH + "F1.png")
    output = hadamard_transform(img)
    image_write(output, OUTPUT_PATH + "H1.png")
    output = cosine_transform(img)
    image_write(output, OUTPUT_PATH + "C1.png")
        #image 2
    img = image_read(INPUT_PATH + "2.png")
    output = fourier_transform(img)
    image_write(output, OUTPUT_PATH + "F2.png")
    output = hadamard_transform(img)
    image_write(output, OUTPUT_PATH + "H2.png")
    output = cosine_transform(img)
    image_write(output, OUTPUT_PATH + "C2.png")

    #step 3
    img = image_read(INPUT_PATH + "3.png")
    r_1 = 30
    r_2 = 60
    r_3 = 100
    output = ideal_low_pass(img,r_1)
    image_write(output, OUTPUT_PATH + "ILP_r1.png")
    output = ideal_low_pass(img,r_2)
    image_write(output, OUTPUT_PATH + "ILP_r2.png")
    output = ideal_low_pass(img,r_3)
    image_write(output, OUTPUT_PATH + "ILP_r3.png")

    output = gaussian_low_pass(img, r_1)
    image_write(output, OUTPUT_PATH + "GLP_r1.png")
    output = gaussian_low_pass(img, r_2)
    image_write(output, OUTPUT_PATH + "GLP_r2.png")
    output = gaussian_low_pass(img, r_3)
    image_write(output, OUTPUT_PATH + "GLP_r3.png")

    output = butterworth_low_pass(img, r_1,2)
    image_write(output, OUTPUT_PATH + "BLP_r1.png")
    output = butterworth_low_pass(img, r_2,2)
    image_write(output, OUTPUT_PATH + "BLP_r2.png")
    output = butterworth_low_pass(img, r_3,2)
    image_write(output, OUTPUT_PATH + "BLP_r3.png")
    # #
    # # step 4
    img = image_read(INPUT_PATH + "3.png")
    r_1 = 30
    r_2 = 50
    r_3 = 100
    output = ideal_high_pass(img, r_1)
    image_write(output, OUTPUT_PATH + "IHP_r1.png")
    output = ideal_high_pass(img, r_2)
    image_write(output, OUTPUT_PATH + "IHP_r2.png")
    output = ideal_high_pass(img, r_3)
    image_write(output, OUTPUT_PATH + "IHP_r3.png")
    #
    output = gaussian_high_pass(img, r_1)
    image_write(output, OUTPUT_PATH + "GHP_r1.png")
    output = gaussian_high_pass(img, r_2)
    image_write(output, OUTPUT_PATH + "GHP_r2.png")
    output = gaussian_high_pass(img, r_3)
    image_write(output, OUTPUT_PATH + "GHP_r3.png")

    output = butterworth_high_pass(img, r_1,2)
    image_write(output, OUTPUT_PATH + "BHP_r1.png")
    output = butterworth_high_pass(img, r_2,2)
    image_write(output, OUTPUT_PATH + "BHP_r2.png")
    output = butterworth_high_pass(img, r_3,2)
    image_write(output, OUTPUT_PATH + "BHP_r3.png")

    #step5
       # part a
    img = image_read(INPUT_PATH + "4.png")
    freq= 50
    bandwidht = 0
    bandwidht_2 = 200
    output =  band_reject_filter(img, freq,bandwidht,bandwidht_2)
    image_write(output, OUTPUT_PATH + "BR1.png")

    output = band_pass_filter(img, freq,bandwidht,bandwidht_2)
    image_write(output, OUTPUT_PATH + "BP1.png")

    #     #part b
    img = image_read(INPUT_PATH + "5.png")
    freq = 50
    bandwidht = 490
    bandwidht_2 = 500
    output = band_reject_filter(img, freq, bandwidht, bandwidht_2)
    image_write(output, OUTPUT_PATH + "BR2.png")

    output = band_pass_filter(img,freq ,bandwidht, bandwidht_2)
    image_write(output, OUTPUT_PATH + "BP2.png")


