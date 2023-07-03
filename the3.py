import math
import os
from sklearn.cluster import KMeans
import cv2
import numpy as np
from matplotlib import pyplot as plt

INPUT_PATH = "./THE3_Images/"
OUTPUT_PATH = "./Outputs/"

image = cv2.imread(INPUT_PATH+'baba.jpeg', cv2.IMREAD_GRAYSCALE)

pseudocolored_image = cv2.applyColorMap(image, cv2.COLORMAP_PINK )

cv2.imwrite(OUTPUT_PATH+'pseudocolored_image.png', pseudocolored_image)
def image_read(img_path,mode ="rgb"):
    if(mode=="gray"):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(img_path)
    return image

def image_write(img, output_path):
    cv2.imwrite(output_path, img)


def closest_non_zero(l, i):
    if l[i] > 0:
        return l[i]

    arr = np.array(l)
    non_zeros = np.nonzero(arr)[0]

    distances = np.abs(non_zeros - i)
    closest_idx = np.min(np.where(distances == np.min(distances)))


    return arr[non_zeros[closest_idx]]

def get_color_map(rgb_image):
    b, g, r = cv2.split(rgb_image)
    r_hist, bins = np.histogram(r, bins=256, range=(0, 256))
    g_hist, bins = np.histogram(g, bins=256, range=(0, 256))
    b_hist, bins = np.histogram(b, bins=256, range=(0, 256))
    r_hist = np.ceil(r_hist / np.amax(r_hist) * 255).astype(int)
    g_hist = np.ceil(g_hist / np.amax(g_hist) * 255).astype(int)
    b_hist = np.ceil(b_hist / np.amax(b_hist) * 255).astype(int)


    mapper = np.zeros((3, 256, 1))
    for i in range(rgb_image.shape[0] ):
        for j in range(rgb_image.shape[1] ):
            value = (int)(0.299 * rgb_image[i][j][2] + 0.587 * rgb_image[i][j][1] + 0.114 * rgb_image[i][j][0])
            if (r_hist[value] > g_hist[value]):
                if (r_hist[value] > b_hist[value]):
                    if (mapper[0][value] < rgb_image[i][j][0]):
                        mapper[0][value] += rgb_image[i][j][0]
                        mapper[1][value] += rgb_image[i][j][1]
                        mapper[2][value] += rgb_image[i][j][2]
                else:
                    if (mapper[2][value] < rgb_image[i][j][2]):
                        mapper[0][value] += rgb_image[i][j][0]
                        mapper[1][value] += rgb_image[i][j][1]
                        mapper[2][value] += rgb_image[i][j][2]
            elif (g_hist[value] > b_hist[value]):
                if (mapper[1][value] < rgb_image[i][j][1]):
                    mapper[0][value] += rgb_image[i][j][0]
                    mapper[1][value] += rgb_image[i][j][1]
                    mapper[2][value] += rgb_image[i][j][2]
            else:
                if (mapper[2][value] < rgb_image[i][j][2]):
                    mapper[0][value] += rgb_image[i][j][0]
                    mapper[1][value] += rgb_image[i][j][1]
                    mapper[2][value] += rgb_image[i][j][2]
    mapper[0] = np.ceil(mapper[0] / np.amax(mapper[0]) * 255).astype(int)
    mapper[1] = np.ceil(mapper[1] / np.amax(mapper[1]) * 255).astype(int)
    mapper[2] = np.ceil(mapper[2] / np.amax(mapper[2]) * 255).astype(int)
    return mapper

def color_images(gray_image,rgb_image):
    mapper = get_color_map(rgb_image)
    img_color = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), rgb_image.dtype)
    for i in range( gray_image.shape[0] ):
        for j in range( gray_image.shape[1] ):
            # distance = (int)(abs(gray_image[i][j+1]-gray_image[i][j] + gray_image[i][j-1]-gray_image[i][j] + gray_image[i-1][j]-gray_image[i][j] + gray_image[i+1][j]-gray_image[i][j])/4)

            img_color[i, j, 0] = closest_non_zero(mapper[0], gray_image[i][j])
            img_color[i, j, 1] = closest_non_zero(mapper[1], gray_image[i][j])
            img_color[i, j, 2] = closest_non_zero(mapper[2], gray_image[i][j])

    # img_color = cv2.bitwise_not(img_color)
    return img_color


def rgb_to_hsi(rgb_image):

    rgb_image = rgb_image.astype(np.float32) / 255.0

    # Convert the image to the HSL color space
    hsl_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HLS)

    # Convert the HSL image to the HSI color space
    hsi_image = np.empty_like(hsl_image)
    hsi_image[:, :, 0] = hsl_image[:, :, 0]
    hsi_image[:, :, 1] = hsl_image[:, :, 2]
    hsi_image[:, :, 2] = (hsl_image[:, :, 1] + hsl_image[:, :, 2]) / 2

    return hsi_image

def plot_hsi_rgb(rgb_image,hsi_image):

    # Get the individual color channels
    b = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    r = rgb_image[:, :, 2]

    i = hsi_image[:, :, 2]
    s = hsi_image[:, :, 1]
    h = hsi_image[:, :, 0]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 7))

    # Plot the red channel
    ax1.imshow(r, cmap='Reds')
    ax1.set_title('Red Channel (RGB)')
    ax1.axis('off')

    # Plot the green channel
    ax2.imshow(g, cmap='Greens')
    ax2.set_title('Green Channel (RGB)')
    ax2.axis('off')

    # Plot the blue channel
    ax3.imshow(b, cmap='Blues')
    ax3.set_title('Blue Channel (RGB)')
    ax3.axis('off')

    # Plot the hue channel
    ax4.imshow(h, cmap='hsv')
    ax4.set_title('Hue Channel (HSI)')
    ax4.axis('off')

    # Plot the saturation channel
    ax5.imshow(s, cmap='gray', vmin=0, vmax=1)
    ax5.set_title('Saturation Channel (HSI)')
    ax5.axis('off')

    # Plot the intensity channel
    ax6.imshow(i, cmap='gray', vmin=0, vmax=1)
    ax6.set_title('Intensity Channel (HSI)')
    ax6.axis('off')

    # Show the plot
    plt.show()




def detect_faces(img, num_of_faces, mode):
    # Convert rgb color to hsv color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Apply filters if necessary
    if (mode == 0):
        # hsv = cv2.GaussianBlur(hsv, (25, 25), 0)
        # hsv = cv2.medianBlur(hsv, 25)
        pass
    elif(mode == 1):
        hsv = cv2.medianBlur(hsv, 5)
    else:
        hsv = cv2.GaussianBlur(hsv, (15, 15), 0)
        hsv = cv2.medianBlur(hsv, 15)
    # Define the range of HSV values for skin color in masks
    masks = [[np.array([5, 80, 240]),np.array([15, 200, 255])],[np.array([0, 70, 130]),np.array([20, 150, 255])],[np.array([0, 70, 130]),np.array([20, 150, 255])]]
    lower_skin = masks[mode][0]
    upper_skin = masks[mode][1]
    # Create the mask image using the inRange function
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Apply the mask using the bitwise_and operator
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    #Morphological operations
    if (mode == 0):
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=3)
        kernel = np.ones((5, 5), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=6)
        # return gray
    elif (mode == 1):
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=8)
        gray = cv2.dilate(gray, kernel, iterations=8)
    else:
        kernel = np.ones((1, 33), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=2)
        gray = cv2.dilate(gray, None, iterations=33)

    #take faces in rectangles
    #find contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #find the num_of_Faces largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_of_faces]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img

def detect_edges(img, k_size, mode):
    if mode == "rgb":
        # Apply the Sobel operator to detect edges in the x and y directions
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)

        # Calculate the gradient magnitude and direction
        magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        direction = np.arctan2(sobel_y, sobel_x)

        # Normalize the gradient magnitude
        magnitude = (magnitude / np.max(magnitude)) * 255

        # Convert the gradient magnitude to an 8-bit image
        magnitude = np.uint8(magnitude)

        # Threshold the gradient magnitude to create a binary edge map
        _, threshold = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "hsi":
        image = img
        # Convert the image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


        # Apply the Sobel operator to detect edges in the x and y directions
        sobel_x = cv2.Sobel(hsv[:,:,2], cv2.CV_64F, 1, 0, ksize=k_size)
        sobel_y = cv2.Sobel(hsv[:,:,2], cv2.CV_64F, 0, 1, ksize=k_size)

        # Calculate the gradient magnitude and direction
        magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        direction = np.arctan2(sobel_y, sobel_x)

        # Normalize the gradient magnitude
        magnitude = (magnitude / np.max(magnitude)) * 255

        # Convert the gradient magnitude to an 8-bit image
        magnitude = np.uint8(magnitude)

        # Threshold the gradient magnitude to create a binary edge map
        _, threshold = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold



if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

       # step 1
    #image 1
    img = image_read(INPUT_PATH + "1_source.png")
    output = detect_faces(img,3,0)
    image_write(output, OUTPUT_PATH + "1_faces.png")
    # image 2
    img = image_read(INPUT_PATH + "2_source.png")
    output = detect_faces(img,5,1)
    image_write(output, OUTPUT_PATH + "2_faces.png")
    # image 3
    img = image_read(INPUT_PATH + "3_source.png")
    output = detect_faces(img,1,2)
    image_write(output, OUTPUT_PATH + "3_faces.png")


       # step 2
    #image 1
    gray_img = image_read(INPUT_PATH + "1.png","gray")
    rgb_img = image_read(INPUT_PATH + "1_source.png")
    output = color_images(gray_img,rgb_img)
    image = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_write(output, OUTPUT_PATH + "1_colored.png")
    # image 2
    gray_img = image_read(INPUT_PATH + "2.png","gray")
    rgb_img = image_read(INPUT_PATH + "2_source.png")
    output = color_images(gray_img,rgb_img)
    image_write(output, OUTPUT_PATH + "2_colored.png")
    # image 3
    gray_img = image_read(INPUT_PATH + "3.png","gray")
    rgb_img = image_read(INPUT_PATH + "3_source.png")
    output = color_images(gray_img,rgb_img)
    image_write(output, OUTPUT_PATH + "3_colored.png")
    # image 4
    gray_img = image_read(INPUT_PATH + "4.png","gray")
    rgb_img = image_read(INPUT_PATH + "4_source.png")
    output = color_images(gray_img,rgb_img)
    image_write(output, OUTPUT_PATH + "4_colored.png")

    # step 2-hsi-rgb
    # image 1
    rgb_img = image_read(OUTPUT_PATH + "1_colored.png")
    hsi_img = rgb_to_hsi(rgb_img)
    plot_hsi_rgb(rgb_img,hsi_img)
    # image 2
    rgb_img = image_read(OUTPUT_PATH + "2_colored.png")
    hsi_img = rgb_to_hsi(rgb_img)
    plot_hsi_rgb(rgb_img, hsi_img)
    # image 3
    rgb_img = image_read(OUTPUT_PATH + "3_colored.png")
    hsi_img = rgb_to_hsi(rgb_img)
    plot_hsi_rgb(rgb_img, hsi_img)
    # image 4
    rgb_img = image_read(OUTPUT_PATH + "4_colored.png")
    hsi_img = rgb_to_hsi(rgb_img)
    plot_hsi_rgb(rgb_img, hsi_img)
    
    
    
    
    
        #step 2-2
    #image 1
    img = image_read(INPUT_PATH + "1_source.png")
    output = detect_edges(img, 17,"rgb")
    image_write(output, OUTPUT_PATH + "rgb1_colored_edges.png")
    # image 2
    img = image_read(INPUT_PATH + "2_source.png")
    output = detect_edges(img, 31, "rgb")
    image_write(output, OUTPUT_PATH + "rgb2_colored_edges.png")
    # image 3
    img = image_read(INPUT_PATH + "3_source.png")
    output = detect_edges(img, 3, "rgb")
    image_write(output, OUTPUT_PATH + "rgb3_colored_edges.png")
    
    img = image_read(INPUT_PATH + "1_source.png")
    output = detect_edges(img, 3, "hsi")
    image_write(output, OUTPUT_PATH + "hsi1_colored_edges.png")
    # image 2
    img = image_read(INPUT_PATH + "2_source.png")
    output = detect_edges(img, 31, "hsi")
    image_write(output, OUTPUT_PATH + "hsi2_colored_edges.png")
    # image 3
    img = image_read(INPUT_PATH + "3_source.png")
    output = detect_edges(img, 3, "hsi")
    image_write(output, OUTPUT_PATH + "hsi3_colored_edges.png")




