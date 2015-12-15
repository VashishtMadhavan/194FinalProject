import cv2
import cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from sklearn.cluster import k_means
from gco_python import pygco
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from sklearn import preprocessing
from sklearn.svm import SVC



"""
class to colorize an image based on the features provided
"""
class Colorizer:
	def __init__(self,featurizer,c=1.0,gam=1.0):
		self.featurizer = featurizer
		self.c = c
		if gam == 1.0:
			self.gam = gam/self.featurizer.pca.n_components_
		else:
			self.gam = gam
		self.num_colors = featurizer.k
		self.svms = [SVC(C=c,gamma=self.gam) for _ in range(self.num_colors)]


	def train(self):
		self.colors = self.featurizer.cluster.cluster_centers_
		self.pca = self.featurizer.pca
		self.labels = self.featurizer.labels
		self.features = self.featurizer.features
		print "Feature Shape..." + str(self.features.shape)
		print "Label Shape..." + str(self.labels.shape)

		for x in range(self.num_colors):
			label = (self.labels == x).astype(np.int32)
			self.svms[x].fit(self.features,label)

	def graph_cut(self,im):
		color_costs = np.zeros((self.num_colors,self.num_colors))
		for x in range(self.num_colors):
			for y in range(self.num_colors):
				color_costs[x][y] = np.linalg.norm(self.colors[x]-self.colors[y])
		color_costs = (color_costs).astype('int32')

		temp = np.hstack(np.array([-1.0*self.distances[i] for i in range(self.num_colors)]))
		new_shape = (im.shape[0],im.shape[1],self.num_colors)
		label_affinity = (100*temp.reshape(new_shape)).astype('int32')

		blurred = cv2.GaussianBlur(im, (0, 0), 3)
		vh = cv2.Sobel(blurred, -1, 1, 0)
		vv = cv2.Sobel(blurred, -1, 0, 1)
		edges = (0.5*vv + 0.5*vh).astype('int32')
		gc = pygco.cut_simple_vh(label_affinity, color_costs, edges, edges, n_iter=10, algorithm='swap')
		return gc


	def color_image(self,im):
		lab = cv2.cvtColor(im, cv.CV_RGB2Lab)
		g = lab[:,:,0]
		features = self.featurizer.get_features(im)

		#distances from the respective margins - can be used as a proxy for probabilities of pixel being a color
		self.distances = np.array([self.svms[i].decision_function(features) for i in range(self.num_colors)])
		print "Done Computing Margins..."
		output = self.graph_cut(g)
		print "Done Getting Graph Cut"
		m,n = output.shape

		output = self.colors[output.reshape(m*n)].reshape(m,n,2)

		return  cv2.cvtColor(cv2.merge((g, np.uint8(output[:,:,0]), np.uint8(output[:,:,1]))), cv.CV_Lab2RGB)



