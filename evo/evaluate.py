import behtree.treegen as tg
import sys
import simulation.bsim as bsim
import simulation.csim as csim
import numpy as np
from multiprocessing import current_process

'''
Set of functions used to evaluate the performance of individuals.
'''

def serial(pop, oldswarm, boxes, genum, timesteps, treecost, field, grid):
	# Evaluate fitness of each individual in population
	for z in range(0, len(pop)):

		swarm = bsim.swarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = oldswarm.origin
		swarm.gen_agents()
		
		env = bsim.map()
		'''
		The map has to be set to the same as passed into the evolution!!!
		'''
		
		#swarm = oldswarm.copy()
		# Decode genome into executable behaviour tree
		print( 'Evaluating Individual: ', z, ' Gen: ', genum)
		bt = tg.tree().decode(pop[z], swarm, boxes)
		tg.tree().ascii_tree(pop[z])
		
		# Set the number of trials per individual to determine fitness
		trials = 1; dur = 0
		score = 0 ; totscore = 0
		record = np.zeros(trials)
		for k in range(trials):


			swarm = bsim.swarm()
			swarm.size = oldswarm.size
			swarm.behaviour = 'none'
			swarm.speed = 0.5
			swarm.gen_agents()
			swarm.grid = grid
			swarm.field = field
			
			env = bsim.map()
			'''
			The map has to be set to the same as passed into the evolution!!!
			'''
			env.bounded = True
			env.env1()
			env.gen()
			swarm.map = env

			boxes = bsim.boxes()
			boxes.set_state('state1')
			boxes.sequence = False
			boxes.radius = 3

			fitness = 0
			t = 0
			found = False
			# IMPORTANT! need to reset behaviours after each run 
			swarm.beacon_set = []
			bt = tg.tree().decode(pop[z], swarm, boxes)
			noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

			# Reset score
			score = 0
			while t <= timesteps and found == False:
				
				bt.tick()
				swarm.iterate(noise[t-1])
				swarm.get_state()
				score = boxes.get_state(swarm,t)
				t += 1
			
			#score = boxes.tot_collected
			print('score = ' , score)
			boxes.reset()
			swarm.reset()
			
		print ('-------------------------------------------------------------------')
		
		maxsize = 300
		fitness = 0

		fitness = score/(len(boxes.boxes))

		fitness = fitness - (len(pop[z].genome)*treecost)
		if fitness < 0: fitness = 0
	
		print ('Individual fitness: %.3f'% fitness)
		pop[z].fitness = fitness
		print ('=================================================================================')


def serial_search(pop, oldswarm, targets, genum, timesteps, treecost, field, grid):
	# Evaluate fitness of each individual in population
	for z in range(0, len(pop)):

		swarm = bsim.swarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = oldswarm.origin
		swarm.gen_agents()
		
		env = bsim.map()
		'''
		The map has to be set to the same as passed into the evolution!!!
		'''
		
		#swarm = oldswarm.copy()
		# Decode genome into executable behaviour tree
		print( 'Evaluating Individual: ', z, ' Gen: ', genum)
		bt = tg.tree().decode(pop[z], swarm, targets)
		tg.tree().ascii_tree(pop[z])
		
		# Set the number of trials per individual to determine fitness
		trials = 1; dur = 0
		score = 0 ; totscore = 0
		record = np.zeros(trials)
		for k in range(trials):


			swarm = bsim.swarm()
			swarm.size = oldswarm.size
			swarm.behaviour = 'none'
			swarm.speed = 0.5
			swarm.gen_agents()
			swarm.grid = grid
			swarm.field = field
			
			env = bsim.map()
			'''
			The map has to be set to the same as passed into the evolution!!!
			'''
			env.bounded = True
			env.map1()
			env.gen()
			swarm.map = env

			targets = bsim.target_set()
			targets.set_state('set1')
			targets.radius = 5
			targets.reset()

			fitness = 0
			t = 0
			found = False
			# IMPORTANT! need to reset behaviours after each run 
			swarm.beacon_set = []
			bt = tg.tree().decode(pop[z], swarm, targets)
			noise = np.random.uniform(-.01,.01,(timesteps, swarm.size, 2))

			# Reset score
			score = 0
			while t <= timesteps and found == False:
				
				bt.tick()
				swarm.iterate(noise[t-1])
				swarm.get_state()
				score = targets.get_state_normal(swarm, t, timesteps)
				t += 1
			
			#score = boxes.tot_collected
			print('score = ' , score)
			targets.reset()
			swarm.reset()
			
		print ('-------------------------------------------------------------------')
		
		maxsize = 300
		fitness = 0

		fitness = score/(len(targets.targets))

		fitness = fitness - (len(pop[z].genome)*treecost)
		if fitness < 0: fitness = 0
	
		print ('Individual fitness: %.3f'% fitness)
		pop[z].fitness = fitness
		print ('=================================================================================')


