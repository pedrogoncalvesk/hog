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


url_dataset = "dataset/"
url_test = "testes/"
url_learning = "treinamento/"
url_sample = "sample/"

pixels_per_cel = 8
cells_per_block = 2
orientations = 9

RUN = ",,"
CREATE = False

url_result = "build/PCP-" + repr(pixels_per_cel) + "-CPB-" + repr(cells_per_block) + "/"


def loop(name, url, create_hog_image):
    print("::" + name + " ::")
    print(time.ctime())
    for file_name in os.listdir(url):
        if file_name == ".DS_Store":
            continue
        print(file_name)
        create_image(file_name, url, create_hog_image)
    print(time.ctime())


def create_image(file_name, url_folder, create_hog_image):
    file_tmp = url_folder + file_name
    file_name = file_tmp[:-4]
    img = imread(file_tmp)
    image = color.rgb2gray(img)

    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=(pixels_per_cel, pixels_per_cel),
                        cells_per_block=(cells_per_block, cells_per_block), visualise=True, feature_vector=True)

    if create_hog_image:
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
            fig.savefig(url_result + url_folder + "comparable_" + file_name.replace("/", "_", 1))
        else:
            fig.savefig(url_result + url_folder + "comparable_" + file_name)

    # save to file
    if file_name.count("/") > 1:
        f = open(url_result + url_folder + "HOG_" + file_name.replace("/", "_", 1) + ".txt", "w")
    else:
        f = open(url_result + url_folder + "HOG_" + file_name + ".txt", "w")

    for item in fd:
        f.write("%s\n" % item)
    f.close()

    plt.close('all')

if not os.path.exists(url_result):
    os.makedirs(url_result)
    os.makedirs(url_result + url_dataset + url_test)
    os.makedirs(url_result + url_dataset + url_test + "comparable_dataset_testes/")
    os.makedirs(url_result + url_dataset + url_test + "HOG_dataset_testes/")
    os.makedirs(url_result + url_dataset + url_learning)
    os.makedirs(url_result + url_dataset + url_learning + "comparable_dataset_treinamento/")
    os.makedirs(url_result + url_dataset + url_learning + "HOG_dataset_treinamento/")
    os.makedirs(url_result + url_sample)
    os.makedirs(url_result + url_sample + "comparable_sample/")
    os.makedirs(url_result + url_sample + "HOG_sample/")

EXECUTED = False

if "RUN" in os.environ:
    RUN = os.environ["RUN"]

if "CREATE" in os.environ:
    TMP_CREATE = os.environ["CREATE"]
    if TMP_CREATE == "true":
        CREATE = True

RUN = RUN.split(",")

if "SAMPLE" in RUN:
    loop("SAMPLE", url_sample, CREATE)
    EXECUTED = True

if "TESTES" in RUN:
    loop("TESTES", url_dataset + url_test, CREATE)
    EXECUTED = True

if "TREINAMENTO" in RUN:
    loop("TREINAMENTO", url_dataset + url_learning, CREATE)
    EXECUTED = True

if EXECUTED:
    print("All Files Created")
else:
    print("You must define the environment variables: RUN and CREATE(optional) ")
    print("\n\nE.g.:   RUN=SAMPLE,TESTES,TREINAMENTO\n        CREATE=true")
