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

class Point:
	def __init__(self,p,dim,id=-1):
		self.coordinates = []
		self.pointList = []
		self.id = id
		self.pointCentroid = 0
		for x in range(0,dim):
			self.coordinates.append(p[x]);
		self.centroid = None
		
class Centroid:
	count = 0
	def __init__(self,point):
		self.point = point
		self.count = Centroid.count
		self.pointList = []
		self.centerPos = []
		self.predictions = []
		self.centerPos.append(self.point)
		self.centroid = None
		Centroid.count += 1

	def update(self,point):
		self.point = point
		self.centerPos.append(self.point)
		
	def addPoint(self,point):
		self.pointList.append(point)
		
	def removePoint(self,point):
		self.pointList.remove(point)
		

class Kmeans:
	def __init__(self, k, pointList, kmeansThreshold, centroidsToRemember, initialCentroids = None):
		self.pointList = []
		self.numPoints = len(pointList)
		self.k = k
		self.initPointList = []
		self.centroidsToRemember = int(k*centroidsToRemember/100)
		print "Centroids to Remember:",self.centroidsToRemember
		self.dim = len(pointList[0])
		self.kmeansThreshold = kmeansThreshold
		self.error = None
		self.errorList = []
		self.closestClusterDistance = {}
		self.centroidDistance = {}

		i = 0
		for point in pointList:
			p = Point(point,self.dim,i)
			i += 1
			self.pointList.append(p)
			self.closestClusterDistance[p.id] = -1
			self.centroidDistance[p.id] = []

		if initialCentroids != None:
			self.centroidList = self.seeds(initialCentroids)
		else:
			self.centroidList = self.selectSeeds(self.k)
		self.mainFunction()

	def selectSeeds(self,k):
		seeds = random.sample(self.pointList, k)
		centroidList = []
		for seed in seeds:
			centroidList.append(Centroid(seed))
		return centroidList

	def seeds(self,initList):
		centroidList = []
		for seed in initList:
			centroidList.append(Centroid(seed))
		return centroidList

	def getDistance(self,point1,point2):
		distance = 0
		for x in range(0,self.dim):
			distance += (point1.coordinates[x]-point2.coordinates[x])**2 
		return (distance)**(0.5)

	def getCentroidInit(self,point):
		minDist = -1
		pos = 0
		for centroid in self.centroidList:
			dist = self.getDistance(point,centroid.point)
			if len(self.centroidDistance[point.id]) < self.centroidsToRemember:
				bisect.insort(self.centroidDistance[point.id], (dist,pos))
			elif self.centroidDistance[point.id][self.centroidsToRemember-1][0] > dist:
				bisect.insort(self.centroidDistance[point.id], (dist,pos))
				del self.centroidDistance[point.id][self.centroidsToRemember]
			if minDist == -1:
				minDist = dist
				closestCentroid = pos
			elif minDist > dist:
				minDist = dist
				closestCentroid = pos
			pos += 1
		return (closestCentroid, minDist)

	def getCentroid(self,point):
		pos = 0
		dist = self.getDistance(point,self.centroidList[point.centroid].point)
		minDist = dist
		closestCentroid = point.centroid
		currCentroid =  point.centroid
		if self.closestClusterDistance[point.id] < dist:
			for x in self.initPointList[point.id]:
				centroid = self.centroidList[x]
				if x != currCentroid:
					dist = self.getDistance(point,centroid.point)
					if minDist > dist:
						minDist = dist
						closestCentroid = x
				pos += 1
		else:
			self.numChange += 1
		self.closestClusterDistance[point.id] = minDist
		return (closestCentroid, minDist)


	def reCalculateCentroid(self):
		pos = 0
		for centroid in self.centroidList:
			zeroArr = []
			for x in range(0,self.dim):
				zeroArr.append(0)
			mean = Point(zeroArr,self.dim)
			for point in centroid.pointList:
				for x in range(0,self.dim):
					mean.coordinates[x] += point.coordinates[x]
			for x in range(0,self.dim):
				try:
					mean.coordinates[x] = mean.coordinates[x]/len(centroid.pointList)
				except:
					mean.coordinates[x] = 0
			centroid.update(mean)
			self.centroidList[pos] = centroid
			pos += 1
	
	def assignPointsInit(self):
		self.initPointList = {}
		for i in range(len(self.pointList)-1,-1,-1):
			temp = self.getCentroidInit(self.pointList[i])
			self.initPointList[self.pointList[i].id] = []
			for l in range(0,self.centroidsToRemember):
				self.initPointList[self.pointList[i].id].append(self.centroidDistance[self.pointList[i].id][l][1])
			centroidPos = temp[0]
			centroidDist = temp[1]
			self.closestClusterDistance[self.pointList[i].id] = centroidDist
			if self.pointList[i].centroid is None:
				self.pointList[i].centroid = centroidPos
				self.centroidList[centroidPos].pointList.append(copy.deepcopy(self.pointList[i]))
			

	def assignPoints(self):
		doneMap = {}
		self.numChange = 0
		for i in range(len(self.centroidList)-1,-1,-1):
			for j in range(len(self.centroidList[i].pointList)-1,-1,-1):
				try:
					a = doneMap[self.centroidList[i].pointList[j].id]
				except:
					doneMap[self.centroidList[i].pointList[j].id] = 1
					temp = self.getCentroid(self.centroidList[i].pointList[j])
					centroidPos = temp[0]
					centroidDist = temp[1]
					if self.centroidList[i].pointList[j].centroid != centroidPos:
						self.centroidList[i].pointList[j].centroid = centroidPos
						self.centroidList[centroidPos].pointList.append(copy.deepcopy(self.centroidList[i].pointList[j]))
						del self.centroidList[i].pointList[j]
		print self.numChange

	def calculateError(self,config):
		error = 0
		for centroid in self.centroidList:
			for point in centroid.pointList:
				error += self.getDistance(point,centroid.point)**2
		return error



	def errorCount(self):
		self.t = threading.Timer(0.5, self.errorCount)
		self.t.start()
		startTime = time.time()
		timeStamp = 0
		if self.error != None:
  			timeStamp = math.log(self.error)
  		endTime = time.time()
  		self.errorList.append(timeStamp)
  		self.ti += 0.5 	

	def mainFunction(self):
		self.iteration = 1
		self.ti = 0.0
		self.errorCount()
		error1 = 2*self.kmeansThreshold+1
		error2 = 0
		iterationNo = 0
		self.currentTime = time.time()
		self.startTime = time.time()
		self.assignPointsInit()
		print "First Step:",time.time() - self.startTime
		while(abs(error1-error2)) > self.kmeansThreshold:
			iterationNo += 1
			self.iteration = iterationNo
			error1 = self.calculateError(self.centroidList)
			self.error = error1
			print "Iteration:",iterationNo,"Error:",error1			
			self.reCalculateCentroid()
			self.assignPoints()
			error2 = self.calculateError(self.centroidList)
			self.error = error2

		self.t.cancel()


def makeRandomPoint(n, lower, upper):
    return numpy.random.normal(loc=upper, size=[lower, n])

# pointList = []
# x = []
# y = []
# c = []
# numPoints = 10000
# dim = 10
# numClusters = 100
# k = 0
# for i in range(0,numClusters):
#     num = int(numPoints/numClusters)
#     p = makeRandomPoint(dim,num,k)
#     k += 5
#     pointList += p.tolist()

# start = time.time()
# #self, k, pointList, kmeansThreshold, predictionThreshold, isPrediction = 0, initialCentroids = None
# config= Kmeans(numClusters,pointList,1000)
# print "Time taken:",time.time() - start
