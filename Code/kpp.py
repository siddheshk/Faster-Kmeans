from __future__ import division
import sys, os
import random
import numpy
import copy
import operator
import time
import threading
import math
import cPickle
import heapq
import itertools
import random
import bisect 
import numpy as np


class Point:
	def __init__(self,p,dim,id=-1):
		self.coordinates = []
		self.pointList = []
		self.id = id
		self.pointCentroid = 0
		for x in range(0,dim):
			self.coordinates.append(p[x]);
		self.centroid = None

class KPP():
	def __init__(self, K, X=None, N=0):
		self.K = K
		if X == None:
			if N == 0:
				raise Exception("If no data is provided, a parameter N (number of points) is needed")
			else:
				self.N = N
				self.X = self._init_board_gauss(N, K)
		else:
			self.X = X
			self.N = len(X)
		self.mu = None
		self.clusters = None
		self.method = None
		self.d = []
		self.D2 = []

	def _dist_from_centers(self):
		cent = self.mu
		X = self.X
		D2 = np.array([np.linalg.norm(x-self.mu[-1])**2 for x in X])
		if len(self.D2) == 0:
			self.D2 = np.array(D2[:])
		else:
			for i in range(len(D2)):
				if D2[i] < self.D2[i]:
					self.D2[i] = D2[i]

	 
	def _choose_next_center(self):
		self.probs = self.D2/self.D2.sum()
		self.cumprobs = self.probs.cumsum()
		print self.cumprobs.shape
		r = random.random()
		ind = np.where(self.cumprobs >= r)[0][0]
		return(self.X[ind])

	def init_centers(self):
		self.mu = random.sample(self.X, 1)
		while len(self.mu) < self.K:
			self._dist_from_centers()
			self.mu.append(self._choose_next_center())

# kplus = KPP(numClusters,X=np.array(pointList))
# kplus.init_centers()
# cList = [Point(x,len(x)) for x in kplus.mu]