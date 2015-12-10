import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv
import skimage.io as skio
from skimage import color
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from sklearn import preprocessing
from sklearn.svm import SVC


"""
class to generate features for a given set of training images
"""
SURF_PARAMS = 20
WINDOW = 20


def convert_lab(train_images):
	lab_ims = []
	for im in train_images:
		t = skio.imread(im)
		q = cv2.cvtColor(t, cv.CV_BGR2Lab)
		lab_ims.append(q)
	return lab_ims


def get_lab(im):
	lab = cv2.cvtColor(im, cv.CV_BGR2Lab)
	return lab


class Featurizer:

	def __init__(self,training_images,k=10,pca_size=32,num_samples=10000):
		self.train = training_images
		self.ntrain = len(training_images)
		self.cluster = None
		self.num_samples = num_samples # the number of samples per training image
		self.k = k
		self.lab_images = convert_lab(training_images)
		self.surf = cv2.SURF()
		self.surf.extended = True
		self.surf.hessianThreshold = 300
		self.pca = PCA(n_components=pca_size) #can change this to adapt the number of features we want to keep
		self.preprocess = preprocessing.MinMaxScaler()


	def compute_k_means(self):
		n,m,k = self.lab_images[0].shape
		pts = self.lab_images[0].reshape(n*m,k)[:,1:]
		for i in range(1,len(self.lab_images)):
			temp = self.lab_images[i].reshape(n*m,k)[:,1:]
			pts = np.vstack((pts,temp))
		cluster_clf = KMeans(n_clusters=self.k)
		cluster_clf.fit(pts)
		print "Cluster Centers..."
		print cluster_clf.cluster_centers_
		self.cluster = cluster_clf

	def discretize_colors(self,im,samples):
		lum = im.reshape(im.shape[0]*im.shape[1],3)[:,:1]
		pixel_arr = im.reshape(im.shape[0]*im.shape[1],3)[:,1:]
		labels = self.cluster.predict(pixel_arr)
		mapped = np.hstack((lum,self.cluster.cluster_centers_[labels]))
		return mapped[samples],labels[samples]

	def surf_features(self,im,samples):
		index = np.array([[x,y] for x in range(im.shape[0]) for y in range(im.shape[1])])
		index = index[samples]
		surf_pts = []
		oct1 = cv2.GaussianBlur(im,(0,0),1)
		oct2 = cv2.GaussianBlur(im,(0,0),2)
		keypoints = np.array([cv2.KeyPoint(i[1], i[0], SURF_PARAMS) for i in index])
		_, des1 = self.surf.compute(im,keypoints)
		_, des2 = self.surf.compute(oct1,keypoints)
		_, des3 = self.surf.compute(oct2,keypoints)
		
		surf_features = np.hstack((np.hstack((des1,des2)),des3))
		return surf_features
	
	def get_patch(self,windowSize,im,pos):
		y_range = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,im.shape[1]))
		x_range = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,im.shape[0]))
		patch = im[x_range[0]:x_range[1],y_range[0]:y_range[1]]
		return patch

	def dft_features(self,im,samples):
		index = np.array([[x,y] for x in range(im.shape[0]) for y in range(im.shape[1])])
		index = index[samples]
		windowSize = WINDOW/2
		l = (2*windowSize + 1)**2
		patches = np.array([self.get_patch(windowSize,im,i) for i in index]) 
		dft_pts = np.array([np.abs(np.fft.fft(patch.flatten())) if patch.shape[0]*patch.shape[1] == l else np.zeros(l) for patch in patches])
		return dft_pts
	
	def local_meanvar_features(self,im,samples):
		index = np.array([[x,y] for x in range(im.shape[0]) for y in range(im.shape[1])])
		index = index[samples]
		windowSize = WINDOW/2
		mean = np.array([np.mean(self.get_patch(windowSize,im,i)) for i in index])
		var = np.array([np.var(self.get_patch(windowSize,im,i)) for i in index])
	
		mean = mean.reshape(len(mean),1)
		var = var.reshape(len(var),1)
		return np.hstack((mean,var))
		
	def compute_features(self):
		for i in range(len(self.lab_images)):
			im = self.lab_images[i]
			lum = im[:,:,0].astype('uint8')
			samples = np.random.choice(lum.shape[0]*lum.shape[1],self.num_samples)
			disc,lab = self.discretize_colors(im,samples)
		
			print "SURF Features..."
			sf = self.surf_features(lum,samples)
			print "DFT Features..."
			dft = self.dft_features(lum,samples)
			print "Meanvar Features..."
			mv = self.local_meanvar_features(lum,samples)
			feat = np.hstack((np.hstack((np.hstack((disc,sf)),dft)),mv))
			if i == 0:
				self.features = feat
				self.labels = lab
			else:
				self.features = np.vstack((self.features,feat))
				self.labels = np.hstack((self.labels,lab))

		self.features = self.preprocess.fit_transform(self.features)
		self.features = self.pca.fit_transform(self.features)
		self.labels = self.labels

	def get_features(self,im):
		lab = get_lab(im)
		lum = lab[:,:,0].astype('uint8')
		samples = np.arange(im.shape[0]*im.shape[1])

		disc,labels = self.discretize_colors(lab,samples)
		srf = self.surf_features(lum,samples)
		dft = self.dft_features(lum,samples)
		mv = self.local_meanvar_features(lum,samples)
		total_features = np.hstack((np.hstack((np.hstack((disc,srf)),dft)),mv))

		#use PCA to compress features to 32
		features = self.preprocess.transform(total_features)
		features = self.pca.transform(features)
		return features

