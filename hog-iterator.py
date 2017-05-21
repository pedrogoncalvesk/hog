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
    image = color.rgb2gray(img)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(pixels_per_cel, pixels_per_cel),
                        cells_per_block=(cells_per_block, cells_per_block), visualise=True, feature_vector=True)

    if createHogImage:
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

        if file_name.count("/") > 1:
            fig.savefig(urlResultado + urlFolder + "comparable_" + file_name.replace("/", "_", 1))
        else:
            fig.savefig(urlResultado + urlFolder + "comparable_" + file_name)

    # save to file
    if file_name.count("/") > 1:
        f = open(urlResultado + urlFolder + "HOG_" + file_name.replace("/", "_", 1) + ".txt", "w")
    else:
        f = open(urlResultado + urlFolder + "HOG_" + file_name + ".txt", "w")

    for item in fd:
        f.write("%s\n" % item)
    f.close()

    plt.close('all')

urlDataset = "dataset/"
urlTeste = "testes/"
urlTreinamento = "treinamento/"
urlSample = "sample/"

pixels_per_cel = 8
cells_per_block = 2

urlResultado = "build/PCP-" + repr(pixels_per_cel) + "-CPB-" + repr(cells_per_block) + "/"

if not os.path.exists(urlResultado):
    os.makedirs(urlResultado)
    os.makedirs(urlResultado + urlDataset + urlTeste)
    os.makedirs(urlResultado + urlDataset + urlTeste + "comparable_dataset_testes/")
    os.makedirs(urlResultado + urlDataset + urlTeste + "HOG_dataset_testes/")
    os.makedirs(urlResultado + urlDataset + urlTreinamento)
    os.makedirs(urlResultado + urlDataset + urlTreinamento + "comparable_dataset_treinamento/")
    os.makedirs(urlResultado + urlDataset + urlTreinamento + "HOG_dataset_treinamento/")
    os.makedirs(urlResultado + urlSample)
    os.makedirs(urlResultado + urlSample + "comparable_sample/")
    os.makedirs(urlResultado + urlSample + "HOG_sample/")

loop("SAMPLE", urlSample, True)
loop("TESTES", urlDataset + urlTeste, False)
loop("TREINAMENTO", urlDataset + urlTreinamento, False)

print("All Files Created");