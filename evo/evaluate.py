import behtree.treegen as tg
import sys
import simulation.asim as asim
import simulation.csim as csim
import numpy as np
from multiprocessing import current_process

'''
Set of functions used to evaluate the performance of individuals.
'''


def serial(pop, oldswarm, targets, genum, timesteps, treecost, states, fields, fitness_maps, grid):
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
		env.bounded = True
		env.room44()
		env.gen()
		swarm.map = env

		#swarm = oldswarm.copy()
		# Decode genome into executable behaviour tree
		print( 'Evaluating Individual: ', z, ' Gen: ', genum)
		bt = tg.tree().decode(pop[z], swarm, targets)
		tg.tree().ascii_tree(pop[z])
		
		# Set the number of trials per individual to determine fitness
		trials = 4; dur = 0
		score = 0 ; totscore = 0
		record = np.zeros(trials)
		for k in range(trials):


			swarm = asim.swarm()
			swarm.size = oldswarm.size
			swarm.behaviour = 'none'
			swarm.speed = 0.5

			if k == 1:
				swarm.origin = np.array([-19, -9])
			if k == 2:
				swarm.origin = np.array([19, -9])
			if k == 3:
				swarm.origin = np.array([-19, 9])
			if k == 4:
				swarm.origin = np.array([19, 9])
			swarm.gen_agents()
			
			env = asim.map()
			'''
			The map has to be set to the same as passed into the evolution!!!
			'''
			env.bounded = True
			env.room44()
			env.gen()
			swarm.map = env


			fitness = 0
			t = 0
			found = False
			# IMPORTANT! need to reset behaviours after each run 
			swarm.beacon_set = []
			bt = tg.tree().decode(pop[z], swarm, targets)

			noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

			swarm.field = fields[0]
			swarm.grid = grid

			targets.fitmap = fitness_maps[0]

			# Reset score
			score = 0

			while t <= timesteps and found == False:
				
				bt.tick()
				swarm.iterate(noise[t-1])
				swarm.get_state()
				score += targets.get_state(swarm, t, timesteps)

				t += 1
				
			print('score = ' , score)

			record[k] = score
			totscore += score
			targets.reset()
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


def cserial(pop, oldswarm, grid, genum, timesteps, treecost):
	# Evaluate fitness of each individual in population

	env = asim.map()
	'''
	The map has to be set to the same as passed into the evolution!!!
	'''
	env.map1()
	env.gen()
	

	field, grid = asim.potentialField_map(env)
	

	for z in range(0, len(pop)):

		swarm = csim.cswarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = oldswarm.origin
		swarm.gen_agents()
		
		env = asim.map()
		'''
		The map has to be set to the same as passed into the evolution!!!
		'''
		env.map1()
		env.gen()
		swarm.map = env

		targets = asim.target_set()
		targets.set_state('uniform')
		targets.radius = 5
		
		targets.reset()

		#swarm = oldswarm.copy()
		# Decode genome into executable behaviour tree
		print( 'Evaluating Individual: ', z, ' Gen: ', genum)
		bt = tg.tree().decode(pop[z], swarm, targets)
		tg.tree().ascii_tree(pop[z])
		
		# Set the number of trials per individual to determine fitness
		trials = 10; dur = 0
		score = 0 ; totscore = 0
		record = np.zeros(trials)

		for k in range(trials):
			
			swarmsize = oldswarm.size
			swarm = csim.cswarm()
			swarm.size = swarmsize
			swarm.speed = 0.5
			swarm.origin = np.array([0, 0])
			swarm.gen_agents()

			grid = csim.gridset()
			grid.distribution = 0.60
			grid.gen()

			targets = asim.target_set()
			targets.radius = 5
			targets.set_state('uniform')
			targets.reset()

			env = asim.map()
			env.empty()
			env.gen()
			swarm.map = env

			swarm.field = field
			swarm.grid = grid

			fitness = 0
			t = 0
			found = False
			# IMPORTANT! need to reset behaviours after each run 
			swarm.beacon_set = []
			bt = tg.tree().decode(pop[z], swarm, targets)

			noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

			while t <= timesteps and found == False:
				t += 1
				
				bt.tick()
				swarm.iterate()
				swarm.check_grid(grid)
				swarm.get_state()
				
			record[k] = swarm.belief
			targets.reset()
			swarm.reset()
			
		print ('-------------------------------------------------------------------')
		
		maxsize = 100
		fitness = 0
		mean = np.mean(record)
		stdev = np.std(record)

		fitness = mean - 2*trials*stdev

		if len(pop[z].genome) > maxsize:
			fitness = 0
		
		print ('Individual fitness: ', fitness)
		pop[z].fitness = fitness
		print ('=================================================================================')




def evaluate_single(ind, oldswarm, targets, timesteps, treecost):
	# Evaluate fitness of each individual in population
	
	# Set the number of trials per individual to determine fitness
	trials = 3; dur = 0
	score = 0 ; totscore = 0
	record = np.zeros(trials)

	for k in range(trials):
		
		swarm = asim.swarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = np.array([0,0])
		swarm.gen_agents()

		env = asim.map()
		'''
		The map has to be set to the same as passed into the evolution!!!
		'''
		env.map1()
		env.gen()
		swarm.map = env

		targets = asim.target_set()
		targets.set_state('set1')
		targets.radius = 5
		targets.reset()

		fitness = 0
		t = 0
		found = False
		# IMPORTANT! need to reset behaviours after each run
		swarm.beacon_set = []

		oldscore = 0

		bt = tg.tree().decode(ind, swarm, targets)
		while t <= timesteps and found == False:
			t += 1
			bt.tick()
			swarm.iterate()
			swarm.get_state()
			
			score = targets.get_state(swarm, t)
			oldscore = score

			#total += (score - oldscore)*((-t/timesteps*3)+1)

		record[k] =  score
		totscore += score
		targets.reset()
		swarm.reset()

	print('record = ' , record)
	
	maxsize = 300
	fitness = 0
	mean = np.mean(record)
	stdev = np.std(record)

	fitness = (mean)/(len(targets.targets))

	fitness = fitness - (len(ind.genome)*treecost)
	#if fitness < 0: fitness = 0

	ind.fitness = fitness
	
	return ind


def parallel(ind, oldswarm, timesteps):
	
	# #For each indivdual generate a new swarm
	print(f"Worker {current_process().pid} running individual")


	#swarm = oldswarm.copy()
	# Decode genome into executable behaviour tree
	
	
	# Set the number of trials per individual to determine fitness
	trials = 1; dur = 0
	score = 0 ; totscore = 0
	record = np.zeros(trials)

	for k in range(trials):
		
		swarm = asim.swarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = np.array([0,0])
		swarm.gen_agents()

		env = asim.map()
		'''
		The map has to be set to the same as passed into the evolution!!!
		'''
		env.map1()
		env.gen()
		swarm.map = env

		targets = asim.target_set()
		targets.set_state('set1')
		targets.radius = 5
		targets.reset()

		fitness = 0
		t = 0
		found = False
		# IMPORTANT! need to reset behaviours after each run
		swarm.beacon_set = []
		bt = tg.tree().decode(ind, swarm, targets)
		

		oldscore = 0
		total = 0

		bt = tg.tree().decode(ind, swarm, targets)
		while t <= timesteps and found == False:
			t += 1
			bt.tick()
			swarm.iterate()
			swarm.get_state()
			
			score = targets.get_state(swarm, t)
			total += (score - oldscore)*((-t/timesteps*3)+1)

			oldscore = score

		record[k] = total

		totscore += score
		targets.reset()
		swarm.reset()

	print('record = ' , record)
	
	maxsize = 300
	fitness = 0
	mean = np.mean(record)
	stdev = np.std(record)

	fitness = (mean-stdev)/(len(targets.targets))

	fitness = fitness - (len(ind.genome)*0.001)
	if fitness < 0: fitness = 0

	ind.fitness = fitness
	
	return ind


def adveserial(popa, popb, swarma, swarmb, targets, genum, timesteps):


	# Two supervisors competing against each other for coverage
	for z in range(0, len(popa)):
		# Decode genome into executable behaviour tree
		print( 'Evaluating Individual: ', z, ' Gen: ', genum)
		bta = tg.tree().decode(popa[z], swarma, targets)
		tg.tree().ascii_tree(popa[z])

		btb = tg.tree().decode(popb[z], swarmb, targets)
		tg.tree().ascii_tree(popb[z])
		
		# Set the number of trials per individual to determine fitness
		trials = 1; dur = 0
		scorea = 0 ; scoreb = 0; totscore = 0
		for k in range(trials):
			fitness = 0
			t = 0
			found = False
			# IMPORTANT! need to reset behaviours after each run 
			
			bta = tg.tree().decode(popa[z], swarma, targets)
			btb = tg.tree().decode(popb[z], swarmb, targets)
			

			while t <= timesteps and found == False:
				t += 1
				bta.tick()
				btb.tick()
				swarma.iterate()
				swarma.get_state()
				swarmb.iterate()
				swarmb.get_state()
				scorea += targets.ad_state(swarma, t)
				scoreb += targets.ad_state(swarmb, t)
				if targets.found == len(targets.targets):
					found = True
			
			#totscore += score
			targets.reset()
			swarma.reset()
			swarmb.reset()
			
		print ('-------------------------------------------------------------------')
		
		maxsize = 300
		fitness = 0
		fitness = scorea/(trials*len(targets.targets))
		fitness = fitness - (len(popa[z].genome)/1000000)
		if fitness < 0: fitness = 0

		popa[z].fitness = fitness

		print ('Individual fitness A: ', fitness)
		popb[z].fitness = fitness
		print ('=================================================================================')

		fitness = 0
		fitness = scoreb/(trials*len(targets.targets))
		fitness = fitness - (len(popb[z].genome)/1000000)
		if fitness < 0: fitness = 0
		
		print ('Individual fitness B: ', fitness)
		popb[z].fitness = fitness
		print ('=================================================================================')

