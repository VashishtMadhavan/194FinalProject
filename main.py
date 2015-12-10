import cv2
import cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from sklearn import preprocessing
from sklearn.svm import SVC
from featurizer import Featurizer,convert_lab
from colorizer import Colorizer



def get_grayscale(image):
	gray = cv2.cvtColor(image, cv.CV_BGR2GRAY)
	return cv2.merge((gray,gray,gray))


if __name__=="__main__":
	training_images = ["images/grass1.jpg","images/grass2.jpg"]
	test_image = skio.imread("images/grass3.jpg")

	#getting the right featurizer
	f = Featurizer(training_images)
	f.compute_k_means()
	print "Getting features..."
	f.compute_features()
	gray_test = get_grayscale(test_image)
	


	#getting the right colorizer
	colorizer = Colorizer(f)
	print "Starting Training of SVMs..."
	colorizer.train()

	#running the experiment
	print "Colorizing Image..."
	colored_image = colorizer.color_image(gray_test)

	print "Grayscale Image"
	skio.imshow(gray_test)
	skio.show()

	print "Colorized Image"
	skio.imshow(colored_image)
	skio.show()
	skio.imsave("results/" + test_image.split("/")[1],colored_image)


