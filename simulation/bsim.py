#!/usr/bin/env python

import time
import random
import sys
import pickle
import string
import numpy as np
import logging
import time
import random
import math
import scipy
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, euclidean

from matplotlib import animation, rc
from IPython.display import HTML


###########################################################################################

class swarm(object):

	def __init__(self):

		self.agents = []
		self.holding = []
		self.boxnum = []

		self.speed = 0.5
		self.headings = []
		self.size = 0
		self.behaviour = 'none'

		self.centermass = [0,0]
		self.spread = 0

		self.param = 3
		self.map = 'none'
		self.beacon_att = np.array([[]])
		self.beacon_rep = np.array([[]])

		self.origin = np.array([0,0])
		self.start = np.array([])

	def gen_agents(self):
		dim = 0.001
		self.holding = np.zeros(self.size)
		self.boxnum = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])
			
	def reset(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.holding = np.zeros(self.size)
		self.boxnum = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = np.zeros(self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])

	def iterate(self):
		global env
		if self.behaviour == 'aggregate':
			aggregate(self, self.param)
		if self.behaviour == 'random':
			random_walk(self)
		if self.behaviour == 'flock':
			flock(self)
		if self.behaviour == 'rot_clock':
			rotate(self, [-2,1])
		if self.behaviour == 'rot_anti':
			rotate(self, [-1,4])
		if self.behaviour == 'avoidance':
			avoidance(self, env)

	def get_state(self):

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)
		totmag = np.sum(mag)
		totpos = np.sum(self.agents, axis=0)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		self.centermass[0] = (totpos[0])/(self.size)
		self.centermass[1] = (totpos[1])/(self.size)

	def copy(self):
		newswarm = swarm()
		newswarm.agents = self.agents[:]
		newswarm.speed = self.speed
		newswarm.size = self.size
		newswarm.behaviour = 'none'
		newswarm.map = self.map.copy()
		#newswarm.beacon_set = self.beacon_set
		return newswarm


# Functions and definitions for set of box objects.

class boxes(object):

	def __init__(self):
		self.boxes = []
		self.radius = 0
		self.picked = []
		self.collected = []
		self.broadcast = []
		self.choice = 0
		self.sequence = False
		self.coverage = 0
		self.collection_size = 7
		self.tot_collected = 0

	def set_state(self, state):

		if state == 'random':
			
			self.boxes = np.random.randint(-24,24, (50,2))
			self.picked = np.zeros(len(self.boxes))
			self.collected = np.zeros(len(self.boxes))
			self.broadcast = np.zeros(len(self.boxes))
			# Randomly set the first box to be collected
			self.broadcast = list(range(0, len(self.boxes)))
			self.choice = random.randint(0, len(self.boxes))



	# Check the state of the boxes 
	def get_state(self, swarm, t):

		score = 0
		# adjacency matrix of agents and boxes
		mag = cdist(swarm.agents, self.boxes)
		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 1)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0

		for n in range(0, len(swarm.agents)):

			# Is the agent holding a box?
			if swarm.holding[n] == 0: 

				# which box is closest?
				closest = np.where(np.amin(mag[n]) == mag[n])
				
				# is box close enough pick up?
				if np.amin(mag[n]) < self.radius and self.picked[closest[0][0]] != 1 and self.collected[closest[0][0]] != 1:
					# box has been picked
					self.picked[closest] = 1

					self.boxes[closest] = swarm.agents[n]

					swarm.boxnum[n] = closest[0][0]
					swarm.holding[n] = 1

			# If agent is holding a box update its position
			if swarm.holding[n] == 1:
	
				self.boxes[int(swarm.boxnum[n])] = swarm.agents[n]
				# Is agent in collection zone
				if -self.collection_size <= swarm.agents[n][0] <= self.collection_size and -self.collection_size <= swarm.agents[n][1] <= self.collection_size:

					self.collected[int(swarm.boxnum[n])] = 1
					swarm.holding[n] = 0

		self.tot_collected += np.sum(self.collected)
		score = np.sum(self.collected)
		return score



	def reset(self):
		pass


# Environment objects which are used within the map class. 

class make_wall(object):

    def __init__(self):

        self.start = np.array([0, 0])
        self.end = np.array([0, 0])
        self.width = 1
        self.hitbox = []


class make_box(object):

    def __init__(self, h, w, origin):

        self.height = h
        self.width = w
        self.walls = []

        self.walls.append(make_wall())
        self.walls[0].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[0].end = [origin[0]+(0.5*w), origin[1]+(0.5*h)]
        self.walls.append(make_wall())
        self.walls[1].start = [origin[0]-(0.5*w), origin[1]-(0.5*h)]; self.walls[1].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]
        self.walls.append(make_wall())
        self.walls[2].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[2].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]
        self.walls.append(make_wall())
        self.walls[3].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[3].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]
        

class make_corridor(object):

    def __init__(self, h, w, orient, origin):

        self.length = h
        self.width = w
        self.origin = origin
        self.walls = []

        if orient == 'vertical':
        
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[0].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[1].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

        if orient == 'horizontal':
        
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*h), origin[1]+(0.5*w)]; self.walls[0].end = [origin[0]+(0.5*h), origin[1]+(0.5*w)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]-(0.5*h), origin[1]-(0.5*w)]; self.walls[1].end = [origin[0]+(0.5*h), origin[1]-(0.5*w)]

class make_doorway(object):

    def __init__(self, l, doorwidth, orient, origin):
        self.walls = []
        if orient == 'vertical':
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0], origin[1]+(0.5*l)]; self.walls[0].end = [origin[0], origin[1]+(0.5*doorwidth)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0], origin[1]-(0.5*doorwidth)]; self.walls[1].end = [origin[0], origin[1]-(0.5*l)]

        if orient == 'horizontal':
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*l), origin[1]]; self.walls[0].end = [origin[0]-(0.5*doorwidth), origin[1]]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]+(0.5*doorwidth), origin[1]]; self.walls[1].end = [origin[0]+(0.5*l), origin[1]]

class make_room(object):

    def __init__(self, h, w, doorwidth, orient, origin):

        self.length = h
        self.width = w
        self.origin = origin
        self.walls = []

        if orient == 'top':
        
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*w), origin[1]-(0.5*h)]; self.walls[0].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[1].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[2].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[2].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[3].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[3].end = [origin[0]-(0.5*doorwidth), origin[1]+(0.5*h)]

            self.walls.append(make_wall())
            self.walls[4].start = [origin[0]+(0.5*doorwidth), origin[1]+(0.5*h)]; self.walls[4].end = [origin[0]+(0.5*w), origin[1]+(0.5*h)]

        if orient == 'bottom':
        
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[0].end = [origin[0]+(0.5*w), origin[1]+(0.5*h)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[1].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[2].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[2].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[3].start = [origin[0]-(0.5*w), origin[1]-(0.5*h)]; self.walls[3].end = [origin[0]-(0.5*doorwidth), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[4].start = [origin[0]+(0.5*doorwidth), origin[1]-(0.5*h)]; self.walls[4].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

        if orient == 'left':
        
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*w), origin[1]-(0.5*h)]; self.walls[0].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[1].end = [origin[0]+(0.5*w), origin[1]+(0.5*h)]

            self.walls.append(make_wall())
            self.walls[2].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[2].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[3].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[3].end = [origin[0]-(0.5*w), origin[1]+(0.5*doorwidth)]

            self.walls.append(make_wall())
            self.walls[4].start = [origin[0]-(0.5*w), origin[1]-(0.5*doorwidth)]; self.walls[4].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]

        if orient == 'right':
        
            self.walls.append(make_wall())
            self.walls[0].start = [origin[0]-(0.5*w), origin[1]-(0.5*h)]; self.walls[0].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[1].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[1].end = [origin[0]+(0.5*w), origin[1]+(0.5*h)]

            self.walls.append(make_wall())
            self.walls[2].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[2].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]

            self.walls.append(make_wall())
            self.walls[3].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[3].end = [origin[0]+(0.5*w), origin[1]+(0.5*doorwidth)]

            self.walls.append(make_wall())
            self.walls[4].start = [origin[0]+(0.5*w), origin[1]-(0.5*doorwidth)]; self.walls[4].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]


# This class contains definitions for different envionments that can be spawned

class map(object):

	def __init__(self):

		self.obsticles = []
		self.force = 0
		self.walls = np.array([])
		self.wallh = np.array([])
		self.wallv = np.array([])
		self.planeh = np.array([])
		self.planev = np.array([])

	def copy(self):
		newmap = map()
		newmap.walls = self.walls[:]
		newmap.wallh = self.wallh[:]
		newmap.wallv = self.wallv[:]
		newmap.planeh = self.planeh[:]
		newmap.planev = self.planev[:]
		newmap.limh = self.limh[:]
		newmap.limv = self.limv[:]
		newmap.gen()
		return newmap

	def gen(self):

		# Perform pre-processing on map object for efficency
		self.walls = np.zeros((2*len(self.obsticles), 2))
		self.wallh = np.zeros((2*len(self.obsticles), 2))
		self.wallv = np.zeros((2*len(self.obsticles), 2))
		self.planeh = np.zeros(len(self.obsticles))
		self.planev = np.zeros(len(self.obsticles))
		self.limh = np.zeros((len(self.obsticles), 2))
		self.limv = np.zeros((len(self.obsticles), 2))

		for n in range(0, len(self.obsticles)):
			# if wall is vertical
			if self.obsticles[n].start[0] == self.obsticles[n].end[0]:
				self.wallv[2*n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
				self.wallv[2*n+1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

				self.planev[n] = self.wallv[2*n][0]
				self.limv[n] = np.array([np.min([self.obsticles[n].start[1], self.obsticles[n].end[1]])-0.5, np.max([self.obsticles[n].start[1], self.obsticles[n].end[1]])+0.5])

			# if wall is horizontal
			if self.obsticles[n].start[1] == self.obsticles[n].end[1]:
				self.wallh[2*n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
				self.wallh[2*n+1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

				self.planeh[n] = self.wallh[2*n][1]
				self.limh[n] = np.array([np.min([self.obsticles[n].start[0], self.obsticles[n].end[0]])-0.5, np.max([self.obsticles[n].start[0], self.obsticles[n].end[0]])+0.5])

			self.walls[2*n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
			self.walls[2*n+1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])


	def env1(self):
		# Bounding Walls ---------------------------------
		box = make_box(50, 50, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

	def env2(self):
		# Bounding Walls ---------------------------------
		box = make_box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		wall = make_wall(); wall.start = [-40, 20]; wall.end = [0,20]; self.obsticles.append(wall)
		wall = make_wall(); wall.start = [40, -20]; wall.end = [0,-20]; self.obsticles.append(wall)

	
		


		
# Swarm behaviours and avoidance below

def avoidance(agents, map):

	size = len(agents)
	# Compute vectors between agents and wall planes
	diffh = np.array([map.planeh-agents[n][1] for n in range(size)])
	diffv = np.array([map.planev-agents[n][0] for n in range(size)])
		
	# split agent positions into x and y arrays
	agentsx = agents.T[0]
	agentsy = agents.T[1]

	# Check intersection of agents with walls
	low = agentsx[:, np.newaxis] >= map.limh.T[0]
	up = agentsx[:, np.newaxis] <= map.limh.T[1]
	intmat = up*low

	# Compute force based vector and multiply by intersection matrix
	Fy = np.exp(-2*abs(diffh) + 5)
	Fy = Fy*diffh*intmat

	low = agentsy[:, np.newaxis] >= map.limv.T[0]
	up = agentsy[:, np.newaxis] <= map.limv.T[1]
	intmat = up*low

	Fx = np.exp(-2*abs(diffv) + 5)
	Fx = Fx*diffv*intmat

	# Sum the forces between every wall into one force.
	Fx = np.sum(Fx, axis=1)
	Fy = np.sum(Fy, axis=1)
	# Combine x and y force vectors
	F = np.array([[Fx[n], Fy[n]] for n in range(size)])
	return F


def rotate(swarm, direction):

	noise = 0.03*np.random.randint(direction[0], direction[1], swarm.size)
	swarm.headings += noise

	# Calculate new heading vector
	gx = 7*np.cos(swarm.headings)
	gy = 7*np.sin(swarm.headings)
	G = np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	A = avoidance(swarm.agents, swarm.map)
	a = G - A

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	swarm.agents += W 


def flock(swarm):

	noise = 0.01*np.random.randint(-100., 100., (swarm.size))
	swarm.headings += noise

	avg = np.sum(swarm.headings)/swarm.size
	swarm.headings += 0.001*avg

	R = 100; r = 2; A = 1; a = 20
	W = np.zeros((swarm.size, 2))
	B = np.zeros((swarm.size, 2))
	
	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)
	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 
	rep = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	rep = np.sum(a, axis =0).T

	# Calculate new heading vector
	gx = 10*np.cos(swarm.headings)
	gy = 10*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	A = avoidance(swarm.agents, swarm.map)
	a = G + A + rep

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)
	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	swarm.agents += W 


def random_walk(swarm):

	alpha = 0.01; beta = 50

	noise = alpha*np.random.randint(-beta, beta, (swarm.size))
	swarm.headings += noise

	# Calculate new heading vector
	gx = 7*np.cos(swarm.headings)
	gy = 7*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	A = avoidance(swarm.agents, swarm.map)
	a = G + A

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	swarm.agents += W 
