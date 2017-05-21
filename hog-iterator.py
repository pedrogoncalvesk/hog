#http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
#https://gurus.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
from skimage.feature import hog
from skimage import color, exposure
from skimage.io import imread
import time
#to crop Image
# def crop():
#     w = img.shape[0]
#     h = img.shape[1]
#     x = 0
#     y =0
#     x1 =0
#     y1 = 0
#     x2 = w
#     y2 = h
#
#     print(img[x][y][0])
#
#     for x in range(0,w):
#         for y in range(0, h):
#             if img[y][x][0] == 0:
#                 print("X1: X:", x," Y:", y)
#                 x1 = x
#                 break
#
#         if x1!= 0:
#             break
#
#     for x in range(w-1, 0, -1):
#         for y in range(0, h):
#             if img[y][x][0] == 0:
#                 print("X2: X:", x," Y:", y)
#                 x2 = x
#                 break
#
#         if x2!= w:
#             break
#
#     for y in range(0, h):
#         for x in range(0,w):
#             if img[y][x][0] == 0:
#                 print("Y1: X:", x," Y:", y)
#                 y1 = y
#                 break
#
#         if y1!= 0:
#             break
#
#     for y in range(h-1, 0, -1):
#         for x in range(0, w):
#             if img[y][x][0] == 0:
#                 print("Y2: X:", x," Y:", y)
#                 y2 = y
#                 break
#
#         if y2!= h:
#             break
#
#
#
#     padding = 5
#     # return img[x1-padding:x2+padding,y1-padding:y2+padding]

def loop(name, url, createHogImage):
    print("::"+ name +" ::")
    print(time.ctime())
    for file_name in os.listdir(url):
        if (file_name == ".DS_Store"):
            continue;
        print(file_name)
        createImages(file_name, url, createHogImage)
    print(time.ctime())


def createImages(file_name, urlFolder, createHogImage):
    sample = urlFolder + file_name
    file_name = sample[:-4]
    img = imread(sample)
    # cropped = crop()
    image = color.rgb2gray(img)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(pixels_per_cel, pixels_per_cel),
                        cells_per_block=(cells_per_block, cells_per_block), visualise=True, feature_vector=True)

    if( createHogImage==True ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')

        # Show Image
        # plt.show()

        fig.savefig(urlResultado + urlFolder + "comparable_" + file_name)

    # save to file
    f = open(urlResultado + urlFolder + "HOG_" + file_name + ".txt", "w")
    for item in fd:
        f.write("%s\n" % item)
    f.close()

    plt.close('all')

#img.shape[0] # (128,128,3)

urlTeste = "dataset/testes/"
urlTreinamento = "dataset/treinamento/"
urlSample = "sample/"

pixels_per_cel = 8
cells_per_block = 2

urlResultado = "PCP-" + repr(pixels_per_cel) + " CPB-" + repr(cells_per_block) + "/"

if not os.path.exists(urlResultado):
    os.makedirs(urlResultado)
    os.makedirs(urlResultado+ urlTeste)
    os.makedirs(urlResultado+ urlTeste+"comparable_testes/")
    os.makedirs(urlResultado+ urlTeste+"HOG_testes/")
    os.makedirs(urlResultado+ urlTreinamento)
    os.makedirs(urlResultado+ urlTreinamento+"comparable_treinamento/")
    os.makedirs(urlResultado+ urlTreinamento+"HOG_treinamento/")
    os.makedirs(urlResultado+ urlSample)
    os.makedirs(urlResultado+ urlSample+"comparable_sample/")
    os.makedirs(urlResultado+ urlSample+"HOG_sample/")

loop("SAMPLE", urlSample, True)
loop("TESTES", urlTeste, False)
loop("TREINAMENTO", urlTreinamento, False)

print("All Files Created")