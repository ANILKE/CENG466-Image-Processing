import os
import matplotlib.pyplot as plt
from skimage import data, color, io, exposure,filters
from skimage.transform import rescale, rotate
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
INPUT_PATH = "./THE1_Images/"
OUTPUT_PATH = "./Outputs/"


def read_image(img_path, rgb=True):
    if (rgb):
        img = io.imread(img_path)
    else:
        img = color.rgb2gray(io.imread(img_path))
    return img


def write_image(img, output_path, rgb=True):
    if (rgb):
        io.imsave(output_path, img)
    else:
        io.imsave(output_path, color.rgb2gray(img))


def extract_save_histogram(img, path):
    plt.title('Histogram')
    plt.subplot()
    plt.hist(img.ravel(), bins=20)
    plt.savefig(path)
    plt.close()


def rotate_image(img, degree=0, interpolation_type="linear"):
    # interpolation type: "linear" or "cubic"
    # degree: 45 or 90

    # rows,cols = img.shape[0:2]
    # cols-1 and rows-1 are the coordinate limits.
    # n = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),degree,0.75)
    # dst = cv2.warpAffine(img,n,(cols,rows))
    if (interpolation_type == "linear"):
        tf_img = rotate(img, degree, True, order=1)
    else:
        tf_img = rotate(img, degree, True, order=3)

    return tf_img


def histogram_equalization(img):
    img_hist_eq = exposure.equalize_hist(img)
    return img_hist_eq


def adaptive_histogram_equalization(img):
    img_hist_eq_adaptive = exposure.equalize_adapthist(img)
    return img_hist_eq_adaptive

def edgeGetter(img):
    edge_roberts = filters.roberts(img)
    edge_sobel = filters.sobel(img)

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                             figsize=(8, 4))

    axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
    axes[0].set_title('Roberts Edge Detection')

    axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
    axes[1].set_title('Sobel Edge Detection')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return edge_roberts
def deblur(img):
    psf = np.ones((5, 5)) / 25
    image = conv2(img, psf, 'same')
    image += 0.1 * image.std() * np.random.standard_normal(image.shape)

    deconvolved, _ = restoration.unsupervised_wiener(image, psf)
    return deconvolved
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    PART1
    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")
    
    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a1_45_cubic.png")
    
    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a1_90_linear.png")
    
    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "cubic")
    write_image(output, OUTPUT_PATH + "a1_90_cubic.png")
    
    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a2_45_linear.png")
    
    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a2_45_cubic.png")
    
    # PART2
    img = read_image(INPUT_PATH + "b1.png", rgb=False)
    extract_save_histogram(img, OUTPUT_PATH + "original_histogram.png")
    equalized = histogram_equalization(img)
    extract_save_histogram(equalized, OUTPUT_PATH + "equalized_histogram.png")
    write_image(equalized, OUTPUT_PATH + "enhanced_image.png")
    # In write image, parameter was "output" variable instead of "equalized", and it was outputting the -
    # rotated image instead of enhanced image, so we changed it into equalized image.
    
    # BONUS
    # Define the following function
    equalized = adaptive_histogram_equalization(img)
    extract_save_histogram(equalized, OUTPUT_PATH + "adaptive_equalized_histogram.png")
    write_image(equalized, OUTPUT_PATH + "adaptive_enhanced_image.png")
    # In write image, parameter was "output" variable instead of "equalized", and it was outputting the -
    # rotated image instead of enhanced image, so we changed it into equalized image.
    
    img = read_image(INPUT_PATH + "anil2.jpeg", rgb=False)
    new_equalized = adaptive_histogram_equalization(img)
    write_image(new_equalized, OUTPUT_PATH + "enhanced_anil.jpeg")
    img = read_image(INPUT_PATH + "yol.jpg", rgb=False)
    equalized = adaptive_histogram_equalization(img)
    deblured=deblur(equalized)
    #new_equalized = edgeGetter(deblured)
    image_rescaled = rescale(img, 0.25, anti_aliasing=False)
    image_resized = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
                           anti_aliasing=True)
    image_downscaled = downscale_local_mean(img, (4, 3))
    write_image(image_downscaled, OUTPUT_PATH + "edgedetected_yol.jpeg")



