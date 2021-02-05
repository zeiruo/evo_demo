import behtree.treegen as tg
import sys
import simulation.asim as asim
import simulation.csim as csim
import numpy as np
from multiprocessing import current_process

'''
Set of functions used to evaluate the performance of individuals.
'''

def serial(pop, oldswarm, targets, genum, timesteps, treecost):
	# Evaluate fitness of each individual in population
	for z in range(0, len(pop)):

		swarm = asim.swarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = oldswarm.origin
		swarm.gen_agents()
		
		env = asim.map()
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


			swarm = asim.swarm()
			swarm.size = oldswarm.size
			swarm.behaviour = 'none'
			swarm.speed = 0.5
			swarm.gen_agents()
			
			env = asim.map()
			'''
			The map has to be set to the same as passed into the evolution!!!
			'''
			env.bounded = True
			env.env1()
			env.gen()
			swarm.map = env

			fitness = 0
			t = 0
			found = False
			# IMPORTANT! need to reset behaviours after each run 
			swarm.beacon_set = []
			bt = tg.tree().decode(pop[z], swarm, targets)

			noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))


			# Reset score
			score = 0
			while t <= timesteps and found == False:
				
				bt.tick()
				swarm.iterate(noise[t-1])
				swarm.get_state()
				score += boxes.get_state(swarm, t)

				t += 1
				
			print('score = ' , score)

			record[k] = score
			totscore += score
			boxes.reset()
			swarm.reset()
			
		print ('-------------------------------------------------------------------')
		
		maxsize = 300
		fitness = 0
		mean = np.mean(record)
		stdev = np.std(record)

		fitness = (mean-stdev)/(len(targets.targets))

		fitness = fitness - (len(pop[z].genome)*treecost)
		#if fitness < 0: fitness = 0
	
		print ('Individual fitness: ', fitness)
		pop[z].fitness = fitness
		print ('=================================================================================')


