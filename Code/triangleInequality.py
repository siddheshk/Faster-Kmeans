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
	def __init__(self, k, pointList, kmeansThreshold, initialCentroids = None):
		self.pointList = []
		self.numPoints = len(pointList)
		self.k = k
		self.initPointList = []
		self.dim = len(pointList[0])
		self.kmeansThreshold = kmeansThreshold
		self.error = None
		self.errorList = []
		self.interClusterDistance = {}
		self.lowerBound = {}
		self.upperBound = {}
		self.minimumClusterDistance = {}
		self.r = {}
		self.oldCentroid = {}
		i = 0
		temp = [0 for x in range(self.k)]
		for point in pointList:
			p = Point(point,self.dim,i)
			i += 1
			self.pointList.append(p)
			self.lowerBound[p.id] = copy.deepcopy(temp)
			self.r[p.id] = False

		for clusters in range(k):
			self.interClusterDistance[clusters] = {}
			self.minimumClusterDistance[clusters] = -1

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

		if point.centroid is not None:
			dist = self.getDistance(point,self.centroidList[point.centroid].point)
			minDist = dist
			closestCentroid = point.centroid
			currCentroid =  point.centroid
		else:
			dist = self.getDistance(point,self.centroidList[pos].point)
			minDist = dist
			closestCentroid = pos
			currCentroid = pos

		self.lowerBound[point.id][closestCentroid] = minDist
		for centroid in self.centroidList:
			if pos != currCentroid:
				#print closestCentroid,pos,currCentroid
				if 0.5*self.interClusterDistance[closestCentroid][pos] < minDist:
					dist = self.getDistance(point,centroid.point)
					self.lowerBound[point.id][pos] = dist
					if minDist > dist:
						minDist = dist
						closestCentroid = pos
			pos += 1
		self.upperBound[point.id] = minDist
		return (closestCentroid, minDist)

	def getCentroid(self,point):
		if self.r[point.id]:
			minDist = self.getDistance(point,self.centroidList[point.centroid].point)
			self.upperBound[point.id] = minDist
			self.r[point.id] = False
		else:
			minDist = self.upperBound[point.id]
		pos = 0
		closestCentroid = point.centroid
		for centroid in self.centroidList:
			if pos != point.centroid:
				if self.upperBound[point.id] > self.lowerBound[point.id][pos]:
					if self.upperBound[point.id] > 0.5*self.interClusterDistance[closestCentroid][pos]:
						if minDist > self.lowerBound[point.id][pos] or minDist > 0.5*self.interClusterDistance[closestCentroid][pos]:
							dist = self.getDistance(point,centroid.point)
							self.lowerBound[point.id][pos] = dist
							if minDist > dist:
								minDist = dist
								closestCentroid = pos
								self.upperBound[point.id] = minDist
			pos += 1
		return (closestCentroid, minDist)


	def reCalculateCentroid(self):
		pos = 0
		for centroid in self.centroidList:
			self.oldCentroid[pos] = copy.deepcopy(centroid.point)
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
	
	def calcInterCluster(self):
		for i in range(0,self.k):
			for j in range(i+1,self.k):
				temp = self.getDistance(self.centroidList[i].point,self.centroidList[j].point)
				self.interClusterDistance[i][j] = temp
				self.interClusterDistance[j][i] = temp
				if self.minimumClusterDistance[i] == -1 or self.minimumClusterDistance[i] > 0.5*temp:
					self.minimumClusterDistance[i] = 0.5*temp
				if self.minimumClusterDistance[j] == -1 or self.minimumClusterDistance[j] > 0.5*temp:
					self.minimumClusterDistance[j] = 0.5*temp

	def assignPointsInit(self):
		self.calcInterCluster()
		for i in range(len(self.pointList)-1,-1,-1):
			temp = self.getCentroidInit(self.pointList[i])
			centroidPos = temp[0]
			centroidDist = temp[1]
			if self.pointList[i].centroid is None:
				self.pointList[i].centroid = centroidPos
				self.centroidList[centroidPos].pointList.append(copy.deepcopy(self.pointList[i]))
			

	def assignPoints(self):
		doneMap = {}
		self.calcInterCluster()
		self.distanceMap = {}
		for x in range(self.k):
			self.distanceMap[x] = self.getDistance(self.oldCentroid[x],self.centroidList[x].point)
		for i in range(len(self.centroidList)-1,-1,-1):
			for j in range(len(self.centroidList[i].pointList)-1,-1,-1):
				try:
					a = doneMap[self.centroidList[i].pointList[j].id]
				except:
					for x in range(self.k):
						self.lowerBound[self.centroidList[i].pointList[j].id][x] = max((self.lowerBound[self.centroidList[i].pointList[j].id][x] - self.distanceMap[x]),0)
					self.upperBound[self.centroidList[i].pointList[j].id] += self.distanceMap[self.centroidList[i].pointList[j].centroid]
					self.r[self.centroidList[i].pointList[j].id] = True
					doneMap[self.centroidList[i].pointList[j].id] = 1
					if self.upperBound[self.centroidList[i].pointList[j].id] > self.minimumClusterDistance[self.centroidList[i].pointList[j].centroid]:
						temp = self.getCentroid(self.centroidList[i].pointList[j])
						centroidPos = temp[0]
						centroidDist = temp[1]
						if self.centroidList[i].pointList[j].centroid != centroidPos:
							self.centroidList[i].pointList[j].centroid = centroidPos
							self.centroidList[centroidPos].pointList.append(copy.deepcopy(self.centroidList[i].pointList[j]))
							del self.centroidList[i].pointList[j]


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
		self.reCalculateCentroid()
		print "First Step:",time.time() - self.startTime
		while(100 * abs(error1-error2)/abs(error1)) > self.kmeansThreshold:
			iterationNo += 1
			self.iteration = iterationNo
			error1 = self.calculateError(self.centroidList)
			self.error = error1
			print "Iteration:",iterationNo,"Error:",error1			
			self.assignPoints()
			self.reCalculateCentroid()
			error2 = self.calculateError(self.centroidList)
			self.error = error2
		self.assignPoints()
		self.reCalculateCentroid()
		error = self.calculateError(self.centroidList)
		self.error = error
		print "Extra Iteration Error:",error
		time.sleep(1)
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
# config1= Kmeans(numClusters,pointList,1000)
# print "Time taken:",time.time() - start
# cList = []
# for centroid in config1.centroidList:
# 	point = centroid.centerPos[0]
# 	cList.append(point)
# start = time.time()
# #self, k, pointList, kmeansThreshold, predictionThreshold, isPrediction = 0, initialCentroids = None
# config2= kmeans.Kmeans(numClusters,pointList,1000,cList)
# print "Time taken:",time.time() - start
