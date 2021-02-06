#!/usr/bin/env python

import time
import random
import sys
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

import simulation.bsim as bsim
import behtree.treegen as tg
import behtree.tree_nodes as tree_nodes
import evo.evaluate as evaluate
import evo.operators as op

from matplotlib import animation, rc
# from IPython.display import HTML


# Top level execution of evolutionary algorithm

if __name__ == '__main__':

	''' Required inputs to evolutionary algorithm:
		
		1. Create swarm object:
			- Define the swarm size.
			- Initialize with behavior 'none'.
			- Set agent speed.
			- Generate an array of agent positions.

		2. Create environment object:
			- Assign the environment object to the swarm.map attribute.

		3. Create the set of targets to search for:
			- Create a target set object and set it's state to match the environment.
			- Set the detection radius.
	'''

	# Set random seed
	#random.seed(1000)
	seed = random.randrange((2**32) - 1)
	random.seed(100)
	np.random.seed(100)

	# Create swarm object
	swarmsize = 10
	swarm = bsim.swarm()
	swarm.size = swarmsize
	swarm.behaviour = 'none'
	swarm.speed = 0.5
	swarm.origin = np.array([0, 0])
	swarm.gen_agents()

	# Create environment
	env = bsim.map()
	'''
	Check that environment is set to the right one! 
	'''
	env.bounded = True
	env.env1() 
	env.gen()
	swarm.map = env

	boxes = bsim.boxes()
	boxes.set_state('random')
	boxes.sequence = False
	boxes.radius = 3

	states = []

	# The test duration for each search attempt
	timesteps = 300

	# Generate and store all field maps
	fields = []
	fitness_maps = []

	#for state in states:

	env = bsim.map()
	env.bounded = True
	env.env1()
	env.gen()

	field, grid = bsim.potentialField_map(env)

	trials = 1
	
	# EVOLUTIONARY SETUP --------------------------------------------------------------------
	'''
		Evolutionary algorithm paramters:
		NGEN - Number of evoltuionary generations.
		popsize - Number of individuals in each population.
		indsize - The maximum depth of trees that are initially generated.
		tournsize - The number of individuals taken in each tournament selection.
		mutrate - Probability of performing a mutation on a node of a tree.
		growrate - Probability of growing a random tree during mutation.
		growdepth - The depth of randomly grown trees.
		hallsize - The number of individuals saved in the hall of fame.
		elitesize - The number of the best individuals saved between generations.  
	'''

	blackboard = {"operators": ['sel','seq'], 
					"opsize": [2,3,4,5,6,7],
					"act_types": ['act','param','env'],
					"actions": ['disperse','north','south','west','east','southeast','southwest','northeast','northwest','random','rot_anti','rot_clock'],
					"dirparam": [10,30,60],
					"rotparam": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09],
					"randparam": [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04],
					"envcontrol": ['attract','repel'],
					"envcontrol_x": [-10,-8,-6,-4,-2,0,2,4,6,8,10],
					"envcontrol_y": [-10,-8,-6,-4,-2,0,2,4,6,8,10],
					"metrics": ['coverage', 'medianx', 'mediany', 'density'],
					"dimx": [-10,-8,-6,-4,-2,0,2,4,6,8,10],
					"dimy": [-10,-8,-6,-4,-2,0,2,4,6,8,10],
					"density": [2,6,10,14,18,22,26,30,34],
					"coverage": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
					}

	NGEN = 25; popsize = 30; indsize = 2
	tournsize = 4; mutrate = 0.15; growrate = 0.5; growdepth = 2
	extinction_prob = -1; deathrate = 0.9
	hall = []; hallsize = 20; newind = 0
	elitesize = 6
	treecost = 0.001
	op_prob = 0.25
	
	selectionNum = popsize - elitesize - newind
	
	# Generate starting population
	pop = []
	generations = []*NGEN
	pop = [tg.individual(tg.tree().make_tree(indsize, blackboard)) for t in range(popsize)]
	
	#  Logging variables
	logfit = []; logpop = []; logavg = []; logmax = []
	stats = {"avgsize": [], "stdsize": [], "meanfit": [], "stdfit": [], "maxfit": []}
	
	# Start evolution!
	for i in range(0, NGEN):
		print ('GEN: ', i) 
		newpop = []		
		
		# Serial execution
		evaluate.serial(pop, swarm, boxes, i, timesteps, treecost, field, grid)

		# Record results -------------------------------------------------------------------------------------------------
		pop.sort(key=lambda x: x.fitness, reverse=True)
		op.log(pop, hall, logpop, logfit, logavg, logmax, i)

		hall = op.hallfame(pop, hall[:], hallsize)
		
		# Generate the next population -----------------------------------------------------------------------------------
		elite = [ind.copy() for ind in pop[0:elitesize]]

		# Remove worst individuals
		newpop = op.tournament(pop[:], tournsize, selectionNum)

		newpop = op.crossover(newpop[:], op_prob)
		
		newpop = op.mutate(newpop[:], mutrate, growrate, growdepth, blackboard)

		pop = []
		pop = list(newpop + elite)

		


		
