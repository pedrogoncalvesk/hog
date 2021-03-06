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

pixels_per_celL = 8
cells_per_block = 1
orientations = 9
file_dimensions = [128, 128]

RUN = ""
CREATE_IMAGE = False

if "PIXELS_PER_CELL" in os.environ:
    pixels_per_celL = int(os.environ["PIXELS_PER_CELL"])

if "CELLS_PER_BLOCK" in os.environ:
    cells_per_block = int(os.environ["CELLS_PER_BLOCK"])

if "ORIENTATIONS" in os.environ:
    orientations = int(os.environ["ORIENTATIONS"])

if "FILE_DIMENSIONS" in os.environ:
    file_dimensions = os.environ["FILE_DIMENSIONS"]
    file_dimensions = map(int, file_dimensions.split("x"))

if "RUN" in os.environ:
    RUN = os.environ["RUN"]

if "CREATE_IMAGE" in os.environ:
    TMP_CREATE = os.environ["CREATE_IMAGE"]
    if TMP_CREATE == "true":
        CREATE_IMAGE = True

url_result = "build/PCP-" + repr(pixels_per_celL) + "-CPB-" + repr(cells_per_block) + "/"


def loop(name, url, create_hog_image):
    print("::" + name + " ::")
    timestamp_start = time.ctime()
    for file_name in os.listdir(url_dataset + url):
        if file_name == ".DS_Store":
            continue
        print(file_name)
        create_image(file_name, url, create_hog_image)
    timestamp_finish = time.ctime()
    return timestamp_start, timestamp_finish


def create_image(file_name, url_folder, create_hog_image):
    file_tmp = url_dataset + url_folder + file_name
    file_name = file_tmp[:-4].replace(url_dataset, "")
    img = imread(file_tmp)
    image = color.rgb2gray(img)

    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=(pixels_per_celL, pixels_per_celL),
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

        fig.savefig(url_result + url_folder + "comparable_" + file_name)

    # Start save to file
    f = open(url_result + url_folder + "HOG_" + file_name + ".txt", "w")

    # Save hog into file
    for item in fd:
        f.write("%s\n" % item)
    f.close()

    plt.close('all')

if __name__ == '__main__':

    if not os.path.exists(url_result):
        os.makedirs(url_result)
        os.makedirs(url_result + url_test)
        os.makedirs(url_result + url_test + "comparable_testes/")
        os.makedirs(url_result + url_test + "HOG_testes/")
        os.makedirs(url_result + url_learning)
        os.makedirs(url_result + url_learning + "comparable_treinamento/")
        os.makedirs(url_result + url_learning + "HOG_treinamento/")

    EXECUTED = False

    timestamp_start = ""
    timestamp_finish = ""

    if RUN:
        if RUN == "TESTES":
            timestamp_start, timestamp_finish = loop("TESTES", url_test, CREATE_IMAGE)
            EXECUTED = True
        elif RUN == "TREINAMENTO":
            timestamp_start, timestamp_finish = loop("TREINAMENTO", url_learning, CREATE_IMAGE)
            EXECUTED = True

    if EXECUTED:
        done = "All files created: START  " + timestamp_start + " FINISH " + timestamp_finish
        print(done)
        f = open(url_result + "config.txt", "w")
        f.write("%s\n" % done)
        f.write("HOG_ORIENTATIONS : %s\n" % orientations)
        f.write("HOG_PIXELS_PER_CELL : %s\n" % pixels_per_celL)
        f.write("HOG_CELLS_PER_BLOCK : %s\n" % cells_per_block)
        f.write("MLP_X_LENGTH : %s\n" % str((file_dimensions[0] / pixels_per_celL) * orientations * (file_dimensions[0] / pixels_per_celL)))
        f.close()
    else:
        print("You must define the environment variables: RUN, CREATE_IMAGE(optional), ")
        print("ORIENTATIONS(default=9), PIXELS_PER_CELL(default=8), CELLS_PER_BLOCK(default=1) ")
        print("\n\nE.g.:   RUN=TESTES,TREINAMENTO"
              "\n        CREATE_IMAGE=true"
              "\n        fi=true")
