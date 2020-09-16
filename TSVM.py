############################################################################################
# TSVM
# 	An implementation of Transductive Support Vector Machine Algorithm
# 	as described in paper (Joachims, T., 1999. Transductive Inference for Text
# 	Classication using Support Vector Machines. In: 16th International
# 	Conference on Machine Learning. Morgan Kaufmann, San Francisco,
# 	CA, pp. 200-209.). We referred to Machine learning (Zhou, Z.-H., 2016. Machine
# 	Learning. Tsinghua University Press, Beijing. 425pp.) and implement TSVM without
# 	unbalanced costs. We also referred to blogs ( https://blog.csdn.net/Horcham/article
# 	/details/86707821?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=
# 	distribute.pc_relevant.none-task and https://blog.csdn.net/FelixWang0515/article/
# 	details/94629025?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=
# 	distribute.pc_relevant.none-task ).
############################################################################################

import numpy as np
import math
from sklearn.svm import SVC

class TSVM:
	def __init__(self,X_l, y, X_u,C_l=1.0,C_u=0.001,kernel='rbf',C=1.0,gamma=1.0):
		'''
		Initialize the model.

		:param X_l: feature values of labeled samples of training dataset
		:param y: labels of labeled samples of training dataset (must be -1 or +1)
		:param X_u: feature values of unlabeled samples of training dataset
		:param C_l: weights of labeled samples (default 1.0, must be a float > 0)
		:param C_u: weights of unlabeled samples (default 0.001, must be a float > 0)
		:param kernel: 'linear' or 'rbf' (default 'rbf')
		:param C: regularization parameter of SVM (default 1, must be a float > 0)
		:param gamma: kernel width for rbf kernel (default 1.0, must be a float > 0)
		'''

		self._X_l=X_l
		self._Y_l=y
		self._X_u=X_u
		self._C_l=C_l
		self._C_u=C_u

		self._kernel=kernel
		self._C=C
		self._gamma=gamma


		if self._kernel=='rbf':
			self._clf=SVC(C=self._C,kernel='rbf',gamma=self._gamma)
		elif self._kernel=='linear':
			self._clf=SVC(C=self._C,kernel='linear')

	def train(self):
		'''
		Train a TSVM.

		'''
		N = len(self._X_l) + len(self._X_u)
		# Initialize weights of labeled and unlabeled samples
		sample_weight = np.ones(N)
		sample_weight[len(self._X_l):] = self._C_u
		# Train a SVM with labeled data
		self._clf.fit(self._X_l,self._Y_l)
		# Get labels of unlabeled samples
		self._Y_u=self._clf.predict(self._X_u)

		X_u_id=np.arange(len(self._X_u))
		self._X=np.vstack([self._X_l, self._X_u])
		self._Y=np.concatenate((self._Y_l,self._Y_u))

		while self._C_u < self._C_l:
			# Train a new SVM with labeled and unlabeled data
			self._clf.fit(self._X, self._Y, sample_weight=sample_weight)
			while True:
				# Get distances from unlabeled samples to the current hyperplane
				distance_Y_u = self._clf.decision_function(self._X_u)
				self._Y_u = self._Y_u.reshape(-1)
				# Calculate function margin
				epsilon = 1 - self._Y_u * distance_Y_u
				# Positive samples
				positive_set, positive_id = epsilon[self._Y_u > 0], X_u_id[self._Y_u > 0]
				# Negative samples
				negative_set, negative_id = epsilon[self._Y_u < 0], X_u_id[self._Y_u < 0]
				positive_max_id = positive_id[np.argmax(positive_set)]
				negative_max_id = negative_id[np.argmax(negative_set)]
				a, b = epsilon[positive_max_id], epsilon[negative_max_id]
				if a > 0 and b > 0 and a + b > 2.0:
					# Switch labels of a pair of unlabeled samples
					Y2[positive_max_id] = Y2[positive_max_id] * -1
					Y2[negative_max_id] = Y2[negative_max_id] * -1
					Y3=np.concatenate((Y1,Y2))
					self._clf.fit(self._X, self._Y, sample_weight=sample_weight)
				else:
					break
			# Renew weights of unlabeled samples
			self._C_u = min(2*self._C_u, self._C_l)
			sample_weight[len(self._X_u):] = self._C_u

	def predict(self,X):
		'''
		Predict labels (-1 or +1) for feature values of samples

		:param X: Feature values of samples
		:return: The predictive labels of X
		'''
		Y = self._clf.predict(X)
		return Y

