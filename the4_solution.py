# Kemal Anil Kekevi 2380608
# Emre Can Koparal 2380673
import numpy as np
from skimage import segmentation, color
from skimage.future import graph
import os

import cv2
import matplotlib.pyplot as plt

from skimage.measure import label

from sklearn.tree import DecisionTreeClassifier, plot_tree

INPUT_PATH = "./THE4_Images/"
OUTPUT_PATH = "./Outputs/"


def image_read(img_path):
    image = cv2.imread(img_path)
    return image


def image_write(img, output_path):
    cv2.imwrite(output_path, img)


def object_counting(img):
    pass




def boundery_overlay(img,segmented_image):
    img_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    edges = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    image_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(image_copy.shape[0]):
        for j in range(image_copy.shape[1]):
            if (edges[i][j] != 0):
                image_copy[i][j][0] = 255
                image_copy[i][j][1] = 255
                image_copy[i][j][2] = 255
    return image_copy


def meanshift_segmentation(img, outputh_pat, parameter_set):

    im_real = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Set the spatial and color bandwidths
    spatial_bandwidth = parameter_set[0]
    color_bandwidth = parameter_set[1]

    # Apply mean shift segmentation
    segmented_image = cv2.pyrMeanShiftFiltering(image, spatial_bandwidth, color_bandwidth)
    _, segmentation_m = cv2.threshold(segmented_image, parameter_set[2], 255, cv2.THRESH_BINARY)
    labels = label(segmentation_m)
    labels_2d =np.zeros((labels.shape[0], labels.shape[1]))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels_2d[i][j] = labels[i][j][0]*1 + labels[i][j][1]*2 +labels[i][j][2]*3
            if labels_2d[i][j] == 0:
                labels_2d[i][j] =255
    if(np.max(labels_2d)>255):
        labels_2d = labels_2d/np.max(labels_2d)*255
    labels_2d =labels_2d.astype(int)
    g = graph.rag_mean_color(img, labels_2d, mode='similarity')

    boundery = boundery_overlay(img,segmentation_m)

    # Create a decision tree classifier
    clf = DecisionTreeClassifier()

    # Train the classifier on the segment labels
    clf.fit(labels_2d.reshape(-1, 1), labels_2d.reshape(-1))

    # Write all 5 images to outputh path

    fig, subs = plt.subplots(1, 5, figsize=(30, 10))

    fig.subplots_adjust(wspace=0.5)

    for i in range(5):
        subs[i].axis('off')

    subs[0].set_title('Image Real')
    subs[1].set_title('Image Segmentation')
    subs[2].set_title('Image Boundery')
    subs[3].set_title('Image Tree')
    subs[4].set_title('Image Region')

    subs[0].imshow(im_real)
    subs[1].imshow(segmentation_m)
    subs[2].imshow(boundery)
    plot_tree(clf, ax=subs[3])
    #subs[3].imshow(img, interpolation='nearest')
    graph.show_rag(labels_2d, g,img, ax=subs[4])
    #subs[4].imshow(region, interpolation='nearest')

    plt.savefig(outputh_pat)


def n_cut_segmentation(img, outputh_pat, parameter_set):
    im_real = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Instantiate a segments array of same shape than image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    labels = segmentation.slic(img, n_segments=parameter_set[0], compactness=parameter_set[1],start_label = 1)
    g = graph.rag_mean_color(img, labels, mode='similarity')
    ncuts_labels = graph.cut_normalized(labels, g)
    ncuts_result = color.label2rgb(ncuts_labels, img, kind='avg')
    boundery = boundery_overlay(img, ncuts_result)
    g = graph.rag_mean_color(img, ncuts_labels, mode='similarity')

    white_image= np.full((img.shape[0],img.shape[1],img.shape[2]),255)


    # Create a decision tree classifier
    clf = DecisionTreeClassifier()

    # Train the classifier on the segment labels
    clf.fit(ncuts_labels.reshape(-1, 1), ncuts_labels.reshape(-1))
    fig, subs = plt.subplots(1, 5, figsize=(30, 10))

    fig.subplots_adjust(wspace=0.5)

    for i in range(5):
        subs[i].axis('off')

    subs[0].set_title('Image Real')
    subs[1].set_title('Image Segmentation')
    subs[2].set_title('Image Boundery')
    subs[3].set_title('Image Tree')
    subs[4].set_title('Image Region')

    subs[0].imshow(im_real)
    subs[1].imshow(ncuts_result)
    subs[2].imshow(boundery)
    plot_tree(clf, ax=subs[3])
    #subs[3].imshow(im_real, interpolation='nearest')
    graph.show_rag(ncuts_labels, g, img, ax=subs[4])
   # subs[4].imshow(im_real, interpolation='nearest')

    plt.savefig(outputh_pat)

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # QUESTION 1

    # QUESTION 2

    # B1
    img = image_read(INPUT_PATH + "B1.jpg")
    # B1.1
    out_path = OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_1.png"
    parameter_set = [40,10,128]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B1.2
    out_path = OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_2.png"
    parameter_set = [20,5,64]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B1.3
    out_path = OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_3.png"
    parameter_set = [30,15,192]
    output = meanshift_segmentation(img, out_path, parameter_set)
    
    #B1 Normalized_Cut
    out_path = OUTPUT_PATH + "B1_algorithm_ncut_parameterset_1.png"
    parameter_set = [400,30]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B1.2
    out_path = OUTPUT_PATH + "B1_algorithm_ncut_parameterset_2.png"
    parameter_set = [100,10]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B1.3
    out_path = OUTPUT_PATH + "B1_algorithm_ncut_parameterset_3.png"
    parameter_set = [300,20]
    output = n_cut_segmentation(img, out_path, parameter_set)
    
     #B2
    img = image_read(INPUT_PATH + "B2.jpg")
    #B2.1
    out_path = OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_1.png"
    parameter_set = [40, 10, 128]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B2.2
    out_path = OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_2.png"
    parameter_set = [20, 5, 64]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B2.3
    out_path = OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_3.png"
    parameter_set = [30, 15, 192]
    output = meanshift_segmentation(img, out_path, parameter_set)
    
    # B2 Normalized_Cut
    out_path = OUTPUT_PATH + "B2_algorithm_ncut_parameterset_1.png"
    parameter_set = [400, 30]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B2.2
    out_path = OUTPUT_PATH + "B2_algorithm_ncut_parameterset_2.png"
    parameter_set = [100, 10]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B2.3
    out_path = OUTPUT_PATH + "B2_algorithm_ncut_parameterset_3.png"
    parameter_set = [300, 20]
    output = n_cut_segmentation(img, out_path, parameter_set)

    # B3
    img = image_read(INPUT_PATH + "B3.jpg")
    # B3.1
    out_path = OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_1.png"
    parameter_set = [40, 10, 128]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B3.2
    out_path = OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_2.png"
    parameter_set = [20, 5, 64]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B3.3
    out_path = OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_3.png"
    parameter_set = [30, 15, 192]
    output = meanshift_segmentation(img, out_path, parameter_set)
    
    # B3 Normalized_Cut
    out_path = OUTPUT_PATH + "B3_algorithm_ncut_parameterset_1.png"
    parameter_set = [400, 30]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B1.2
    out_path = OUTPUT_PATH + "B3_algorithm_ncut_parameterset_2.png"
    parameter_set = [100, 10]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B1.3
    out_path = OUTPUT_PATH + "B3_algorithm_ncut_parameterset_3.png"
    parameter_set = [300, 20]
    output = n_cut_segmentation(img, out_path, parameter_set)



   # B4
    img = image_read(INPUT_PATH + "B4.jpg")
    #B4.1
    out_path = OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_1.png"
    parameter_set = [40,10,128]
    output = meanshift_segmentation(img, out_path, parameter_set)
    # B4.2
    out_path = OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_2.png"
    parameter_set = [20, 5, 64]
    output = meanshift_segmentation(img, out_path, parameter_set)
    #B4.3
    out_path = OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_3.png"
    parameter_set = [30,15,192]
    output = meanshift_segmentation(img, out_path, parameter_set)
    
    #B4 Normalized_Cut
    out_path = OUTPUT_PATH + "B4_algorithm_ncut_parameterset_1.png"
    parameter_set = [400,30]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B4.2
    out_path = OUTPUT_PATH + "B4_algorithm_ncut_parameterset_2.png"
    parameter_set = [100,10]
    output = n_cut_segmentation(img, out_path, parameter_set)
    # B4.3
    out_path = OUTPUT_PATH + "B4_algorithm_ncut_parameterset_3.png"
    parameter_set = [300,20]
    output = n_cut_segmentation(img, out_path, parameter_set)

main()