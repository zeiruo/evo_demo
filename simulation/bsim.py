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
import simulation.environment as environment



###########################################################################################



class swarm(object):

	def __init__(self):

		self.agents = []
		self.speed = 0.5
		self.size = 0
		self.behaviour = 'none'

		self.centroids = []
		self.centermass = [0,0]
		self.median = [0,0]
		self.upper = [0,0]
		self.lower  = [0,0]
		self.spread = 0


		self.field = []
		self.grid = []

		self.param = 3
		self.map = 'none'
		self.beacon_att = np.array([[]])
		self.beacon_rep = np.array([[]])

		self.origin = np.array([0,0])
		self.start = np.array([])

		self.holding = []
		self.boxnum = []

		self.headings = []
		self.shadows = []
		


	def gen_agents(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.holding = np.zeros(self.size)
		self.boxnum = np.zeros(self.size)
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2) + self.origin[0], 0 + self.origin[1]])

		self.shadows = np.zeros((3,swarm.size,2))
	
	def gen_agents_uniform(self, env):

		dim = 0.001
		self.dead = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		
		x = np.random.uniform(-env.dimensions[1]/2, env.dimensions[1]/2, self.size)
		y = np.random.uniform(-env.dimensions[0]/2, env.dimensions[0]/2, self.size)

		self.agents = np.stack((x,y), axis = 1)
		self.shadows = np.zeros((3,self.size,2))

	def reset(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.holding = np.zeros(self.size)
		self.boxnum = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = np.zeros(self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])

		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
	

	def iterate(self, noise):
		global env
		if self.behaviour == 'aggregate':
			aggregate(self, self.param, noise)
		if self.behaviour == 'flocking':
			flocking(self, self.param, noise)
		if self.behaviour == 'random':
			random_walk(self, self.param)
		if self.behaviour == 'rot_clock':
			rotate(self, [-2,2], self.param)
		if self.behaviour == 'rot_anti':
			rotate(self, [-1,3], self.param)
		if self.behaviour == 'disperse':
			dispersion(self, np.array([0,0]), self.param, noise)
		if self.behaviour == 'north':
			dispersion(self, np.array([0,1]), self.param,noise)
		if self.behaviour == 'south':
			dispersion(self, np.array([0,-1]), self.param,noise)
		if self.behaviour == 'west':
			dispersion(self, np.array([-1,0]), self.param,noise)
		if self.behaviour == 'east':
			dispersion(self, np.array([1,0]), self.param,noise)
		if self.behaviour == 'northwest':
			dispersion(self, np.array([-1,1]), self.param,noise)
		if self.behaviour == 'northeast':
			dispersion(self, np.array([1,1]), self.param,noise)
		if self.behaviour == 'southwest':
			dispersion(self, np.array([-1,-1]), self.param,noise)
		if self.behaviour == 'southeast':
			dispersion(self, np.array([1,-1]), self.param,noise)
		if self.behaviour == 'avoidance':
			avoidance(self, env)

	def get_state(self):

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)
		totmag = np.sum(mag)
		#totpos = np.sum(self.agents, axis=0)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		# self.centermass[0] = (totpos[0])/(self.size)
		# self.centermass[1] = (totpos[1])/(self.size)
		self.median = np.median(self.agents, axis = 0)
		# self.upper = np.quantile(self.agents, 0.75, axis = 0)
		# self.lower = np.quantile(self.agents, 0.25, axis = 0)

		# Shift shadows
		for n in range(len(self.shadows)):
			
			self.shadows[len(self.shadows)-n] = self.shadows[len(self.shadows)-n-1]
		self.shadows[0] = swarm.agents

	def copy(self):
		newswarm = swarm()
		newswarm.agents = self.agents[:]
		newswarm.speed = self.speed
		newswarm.size = self.size
		newswarm.behaviour = 'none'
		swarm.origin = self.origin
		newswarm.map = self.map.copy()
		newswarm.field = self.field
		newswarm.grid = self.grid
		#newswarm.beacon_set = self.beacon_set
		return newswarm


class target_set(object):

	def __init__(self):
		self.targets = []
		self.radius = 0
		self.found = 0
		self.coverage = 0
		self.old_state = np.zeros(len(self.targets))
		self.fitmap = []

	def set_state(self, state):


		if state == 'set1':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5],[15,-5],[25,-5],[35,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'set2':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-25,25], [-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-25,15], [-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5], [-25,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'set3':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-25,25], [-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-25,15], [-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5], [-25,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5], [15,-5], [25,-5], [35,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'set4':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-25,25],[-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-25,15],[-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5],[-25,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5],[15,-5],[25,-5],[35,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'brlset':

			x = np.arange(-72.5, 74.9, 2.5)
			y = np.arange(-37.5, 39.9, 2.5)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == 'minimap':

			x = np.arange(-17.5, 19.9, 2.5)
			y = np.arange(-17.5, 19.9, 2.5)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == 'minismall':

			x = np.arange(-9, 9.9, 1)
			y = np.arange(-9, 9.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == 'miniwide':

			x = np.arange(-19, 19.9, 1)
			y = np.arange(-9, 9.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1
		
		
		if state == 'uniform':

			x = np.arange(-40, 45, 3)
			y = np.arange(-40, 45, 3)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1


	def fitness_map(self, map, swarm, timesteps):

		granularity = 1

		# x = np.arange(-72.5,74.9,granularity)
		# y = np.flip(np.arange(-37.5,39.9,granularity))

		x = np.arange(-19,19.9,granularity)
		y = np.flip(np.arange(-9,9.9,granularity))

		pos = np.zeros((len(y),len(x)))


		swarm.behaviour = 'random'
		swarm.param = 0.01

		total_nodes = len(x)*len(y)


		trials = 1

		noise = np.random.uniform(-.1,.1,(trials*timesteps, swarm.size, 2))

		t = 0
		while t <= trials*timesteps:

			if t%timesteps == 0:
				swarm.gen_agents()

			swarm.iterate(noise[t-1])
			swarm.get_state()
			
			# Check intersection of agents with targets
			mag = cdist(self.targets, swarm.agents)
			dist = mag <= granularity

			# For each target sum the number of detections
			total = np.sum(dist, axis = 1)

			# Add the new detections to an array of the positions
			for n in range(len(self.targets)):

				# row = int((self.targets[n][1]+39)/granularity)
				# col = int((self.targets[n][0]+74)/granularity)

				row = int((self.targets[n][1]+9)/granularity)
				col = int((self.targets[n][0]+19)/granularity)

				if total[n] >= 1:
					pos[row][col] += 1
			
			t += 1

			sys.stdout.write("Fit map progress: %.2f   \r" % (100*t/(trials*timesteps)) )
			sys.stdout.flush()

		m = np.max(pos)
		pos = pos/(trials*timesteps)
		pos = pos/np.max(pos)


		# Visualize the probability heatmap
		# plt.imshow(pos, origin='lower')
		# plt.colorbar()
		# plt.show()

		return pos


	def get_state(self, swarm, t, timesteps):

		now = time.time()

		score = 0
		# adjacency matrix of agents and targets
		mag = cdist(swarm.agents, self.targets)

		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 0)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0
		# Check detection against previous state. If a target is already found return false.
		updated = np.logical_or(detected, self.old_state) 
		
		# Accumulate scores for each target found
		# Tracks the total targets found so far. Not this iteration.
		score = np.sum(updated)
		self.coverage = score/len(self.targets)	

	
		# How many targets were found this iteration.
		found = np.logical_xor(detected, self.old_state)*detected

		score = 0

		# Determine score based on decay of target rewards.

		# Get indices of found targets
		found = np.where(found == True)
		
		for ind in found[0]:

			# row = int((self.targets[ind][1]+39)/2.5)
			# col = int((self.targets[ind][0]+74)/2.5)

			row = int((self.targets[ind][1]+9)/1)
			col = int((self.targets[ind][0]+9)/1)

			# Find the decay constant for the target.
			decay = self.fitmap[row][col]
			#value = (1 + (decay)*(-t/timesteps)))
			#score += (1 + (decay*(-t/timesteps)))
			score += np.exp(3*((-t*decay)/timesteps))

		self.old_state = updated
		return score

	def get_state_normal(self, swarm, t, timesteps):

		# Get the target state without reward decay

		now = time.time()

		score = 0
		# adjacency matrix of agents and targets
		mag = cdist(swarm.agents, self.targets)

		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 0)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0
		# Check detection against previous state. If a target is already found return false.
		updated = np.logical_or(detected, self.old_state) 
		
		# Accumulate scores for each target found
		# Tracks the total targets found so far. Not this iteration.
		score = np.sum(updated)
		self.coverage = score/len(self.targets)	



		self.old_state = updated
		return score

	def ad_state(self, swarm, t):

		score = 0
		# adjacency matrix of agents and targets
		mag = cdist(swarm.agents, self.targets)

		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 0)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0
		# Check detection against previous state
		# check which new targets were found
		new = np.logical_and(np.logical_xor(detected, self.old_state), detected) 

		updated = np.logical_or(detected, self.old_state) 
		self.old_state = updated
		score = np.sum(new)
		self.coverage = np.sum(updated)/len(self.targets)	

		return score


	def reset(self):
		self.old_state = np.zeros(len(self.targets))



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

		if state == 'state1':

			self.boxes = np.array([[-15,-8],[-10,4],[-18,7],[5,-10],[-18,20],[-16,20],[-18,18]
									,[-5,18],[-7,18],[-9,18],[-5,20],[-5,22],[-7,20],[-7,22]
									,[14,-5],[16,-5],[18,-5],[20,-5],[22,-5],[14,-7],[16,-7]
									,[18,-7],[20,-7],[22,-7],[14,-9],[16,-9],[18,-9],[20,-9]
									,[20,-20],[10,-20],[21,-17]
									,[-22,-20],[-21,-18],[5,12],[-21,-14],[0,10],[-21,1]
									,[-20,-22],[-18,-22],[-16,-20],[-14,-21],[-11,-20],[-5,-21],[1,-20]
									,[15,20],[10,12],[13,17],[19,10],[20,20],[22,17],[7,20]
									,[-12,-12],[-5,-10],[-16,2]])
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

	def empty(self):

		#box = make_box(50, 50, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		self.dimensions = [50, 50]



	def env1(self):
		# Bounding Walls ---------------------------------
		box = make_box(50, 50, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = make_wall(); wall.start = [25, 0]; wall.end = [10,0]; self.obsticles.append(wall)
		wall = make_wall(); wall.start = [10, 0]; wall.end = [10,-15]; self.obsticles.append(wall)
		# wall = make_wall(); wall.start = [10, -15]; wall.end = [15,-15]; self.obsticles.append(wall)

		# upper room
		wall = make_wall(); wall.start = [-25, 14]; wall.end = [-15,14]; self.obsticles.append(wall)
		wall = make_wall(); wall.start = [-10, 14]; wall.end = [0,14]; self.obsticles.append(wall)
		wall = make_wall(); wall.start = [0, 14]; wall.end = [0,25]; self.obsticles.append(wall)

		wall = make_wall(); wall.start = [12.5, 25]; wall.end = [12.5,14]; self.obsticles.append(wall)


		wall = make_wall(); wall.start = [-17, -17]; wall.end = [-5,-17]; self.obsticles.append(wall)
		wall = make_wall(); wall.start = [-17, -17]; wall.end = [-17,-10]; self.obsticles.append(wall)



	def env2(self):
		# Bounding Walls ---------------------------------
		box = make_box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		wall = make_wall(); wall.start = [-40, 20]; wall.end = [0,20]; self.obsticles.append(wall)
		wall = make_wall(); wall.start = [40, -20]; wall.end = [0,-20]; self.obsticles.append(wall)

	def map1(self):

		self.dimensions = [80,80]

		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		room = environment.room(20, 20, 10, 'top', [0, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 20, 7, 'bottom', [0, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 30, 10, 'bottom', [25, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		#doorway = environment.doorway(30, 7, 'horizontal', [25, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		wall = environment.wall(); wall.start = [10,10]; wall.end = [40,10];
		self.obsticles.append(wall)

		box = environment.box(3, 3, [20, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [30, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [20, -10]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [30, -10]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		doorway = environment.doorway(30, 7, 'horizontal', [25, -20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(30, 7, 'vertical', [10, -25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		doorway = environment.doorway(30, 7, 'horizontal', [-25, -10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		room = environment.room(30, 10, 7, 'right', [-35, -25]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		box = environment.box(15, 3, [-2, -25]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(15, 3, [-18, -25]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(30, 5, [-25, 15]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

	
		


		
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


def potentialField_map(env):

	# Set granularity of field map
	granularity = 0.5

	x = np.arange(-75, 74.9, granularity)
	y = np.arange(-40, 39.9, granularity)
	positions = np.zeros((len(x)*len(y), 2))

	count = 0
	for k in y:
		for j in x:
			positions[count][0] = j 
			positions[count][1] = k
			count += 1

	size = len(positions)
	# Compute vectors between agents and wall planes
	
	diffh = np.array([env.planeh-positions[n][1] for n in range(size)])
	diffv = np.array([env.planev-positions[n][0] for n in range(size)])
	
	# split agent positions into x and y arrays
	agentsx = positions.T[0]
	agentsy = positions.T[1]

	# Check intersection of agents with walls
	low = agentsx[:, np.newaxis] >= env.limh.T[0]
	up = agentsx[:, np.newaxis] <= env.limh.T[1]
	intmat = up*low

	# For larger environments
	A = 10; B = 10
	# For smaller environments
	#A = 2; B = 5

	# Compute force based vector and multiply by intersection matrix
	Fy = np.exp(-A*np.abs(diffh) + B)*diffh*intmat
	
	low = agentsy[:, np.newaxis] >= env.limv.T[0]
	up = agentsy[:, np.newaxis] <= env.limv.T[1]
	intmat = up*low
	now = time.time()
	Fx = np.exp(-A*np.abs(diffv) + B)*diffv*intmat

	# Sum the forces between every wall into one force.
	Fx = np.sum(Fx, axis=1)
	Fy = np.sum(Fy, axis=1)
	# Combine x and y force vectors

	F = np.stack((Fx, Fy), axis = 1)

	return F, positions

def fieldmap_avoidance(swarm):

	F = np.zeros((swarm.size, 2))

	x = np.arange(-75,74.8,0.5)
	y = np.arange(-40,39.9,0.5)

	f = swarm.field.reshape((len(y),len(x),2))

	x = np.round(2*swarm.agents.T[0])/2
	y = np.round(2*swarm.agents.T[1])/2

	inx = np.round(2*(y+40))
	iny = np.round(2*(x+75))

	for n in range(swarm.size):
		# Take agent position and find position in grid
		F[n] = f[int(inx[n])][int(iny[n])]
	
	return F


def dispersion(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	A = fieldmap_avoidance(swarm)

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W

def continuous_boundary(agents, map):

	# Check if agent passes wall bounds. i.e. does agent intersect with area.

	# If yes, which side has the agent passed through?

	# Mirror agent back around to opposite wall.

	# split agent positions into x and y arrays
	agentsx = agents.T[0]
	agentsy = agents.T[1]

	# Set boundary size relative to environment dimensions
	scale = 1

	# Check left and right boundaries
	right = agentsx >=  scale*(map.dimensions[1]/2)
	left = agentsx <= -scale*(map.dimensions[1]/2)

	agentsx += -scale*(map.dimensions[1])*right
	agentsx += scale*(map.dimensions[1])*left

	# Check top and bottom boundaries
	top = agentsy >=  scale*(map.dimensions[0]/2)
	bottom = agentsy <= -scale*(map.dimensions[0]/2)

	agentsy += -scale*(map.dimensions[0])*top
	agentsy += scale*(map.dimensions[0])*bottom

	agents = np.stack((agentsx, agentsy), axis = 1)	

	return agents


def flocking(swarm, param, noise):

	R = 30; r = 3.5; A = 10.5; a = 5.5

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Determine headings
	nearest = mag <= 2

	# n x n matrix of headings of agents which are adjacent
	neighbour_headings = swarm.headings*nearest

	# Sum headings for each agent
	neighbour_headings_tot = np.sum(neighbour_headings, axis = 1)

	# average by number of neighbours

	new_headings = neighbour_headings_tot/(np.sum(nearest, axis = 1))

	# average headings with neighbours
	swarm.headings =  (new_headings + 0.01*np.random.randint(-10,11, swarm.size))

	# Calculate new heading vector
	strength = 10
	gx = strength*np.cos(swarm.headings)
	gy = strength*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#Avoid = fieldmap_avoidance(swarm)
	
	repel = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	repel = np.sum(repel, axis = 0).T

	attract = A*a*np.exp(-mag/a)[:,np.newaxis,:]*diff/(swarm.size-1)	
	attract = np.sum(attract, axis = 0).T

	total = 0
	total +=  noise + repel + G - attract
	
	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)

	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W 
	swarm.agents = continuous_boundary(swarm.agents, swarm.map)
	

	


def aggregate(swarm, param, noise):
	
	R = param; r = 3.5; A = 6.5; a = 7.5

	#noise = np.random.uniform(-.1, .1, (swarm.size, 2))

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	Avoid = fieldmap_avoidance(swarm)
	#B = beacon(swarm)
	
	repel = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	repel = np.sum(repel, axis = 0).T

	attract = A*a*np.exp(-mag/a)[:,np.newaxis,:]*diff/(swarm.size-1)	
	attract = np.sum(attract, axis = 0).T

	total = 0
	total += Avoid + noise + repel - attract
	
	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W 
	


def rotate(swarm, direction, param):

	noise = param*np.random.randint(direction[0], direction[1], swarm.size)
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 2; r = 2; A = 1; a = 20
	# Compute euclidean distance between agents
	# mag = cdist(swarm.agents, swarm.agents)
	# # Compute vectors between agents
	# diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 
	# a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	# a = np.sum(a, axis =0).T

	a = np.zeros((swarm.size,2))
	B = np.zeros((swarm.size, 2))
	#B = beacon(swarm)
	A = avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	swarm.agents += W 


def random_walk(swarm, param):

	alpha = 0.01; beta = 50

	noise = param*np.random.randint(-beta, beta, (swarm.size))
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 20; r = 2; A = 1; a = 20	
	
	a = np.zeros((swarm.size, 2))

	B = np.zeros((swarm.size, 2))
	#B = beacon(swarm)
	A = avoidance(swarm.agents, swarm.map)
	a += A + G + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W

