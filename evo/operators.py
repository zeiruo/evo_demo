
import pickle
import random
import numpy as np
from behtree.treegen import individual
from behtree.treegen import Operator
from behtree.treegen import Action
from behtree.treegen import Condition
from behtree.treegen import Param
from behtree.treegen import Env_control

import behtree.treegen as tg
'''
Evolutionary operators for manipulating trees.

'''

def crossover(offspring, op_prob):


	newpop = []
	choice = list(range(0, len(offspring)))
	parents = list()

	# Check if there is an even number of indivduals

	while len(choice) != 0:
		# Reset parent and children lists
		branch = []; children = []; point = []
		parents = list()

		if len(choice) == 1:
			newpop.append(offspring[choice[0]])
			choice.remove(choice[0])
		else:
			# Select parents from list of possible choices
			for b in range(0, 2):
				sel = random.randint(0,len(choice)-1)
				parents.append(offspring[choice[sel]].copy())
				# print 'seleeeee ',sel 
				choice.remove(choice[sel])
			
			# Choose crossover points
			for c in range(0, 2):
				hasop = False
				# Check whether tree has operator
				for d in range(1, len(parents[c].genome)):
					if type(parents[c].genome[d]) is Operator:
						hasop = True

				# Pick an operator node as the crossover point with probability op_prob.
				if random.random() <= op_prob and hasop == True:
					# Choose operator
					p = random.randint(1,len(parents[c].genome)-1)
					while type(parents[c].genome[p]) is not Operator:
						 p = random.randint(1,len(parents[c].genome)-1)
					#opsum += 1
				else:
					# Choose leaf node
					p = random.randint(1,len(parents[c].genome)-1)
					while type(parents[c].genome[p]) is Operator:
						 p = random.randint(1,len(parents[c].genome)-1)
					#leafsum += 1
				point.append(p)
			
			branch.append(parents[0].genome[point[0]])
			branch.append(parents[1].genome[point[1]])

			children.append(parents[0].copy())
			children.append(parents[1].copy())

			flat = []
			# Flatten genome
			for z in range(0,2):
				for x in range(0,len(children[z].genome)):
					if type(children[z].genome[x]) is list:
						for y in range(0,len(children[z].genome[x])):
							flat.append(children[z].genome[x][y])
					else:
						flat.append(children[z].genome[x])
				children[z].genome = flat
				# print '\nflattended ind, ' ,flat
				flat = []

			subtree = [[],[]]

			for i in range(0,2):
				# Build the subtree for crossover
				if type(branch[i]) is Operator:

					finished = False
					treepos = []

					# Add new counter for children
					treepos.append(children[i].genome[point[i]].size)
					subtree[i].append(children[i].genome[point[i]])
					n = (point[i]-1) + 1 
					while finished != True:
						n += 1
						# End current operator if max children is reached.
						if treepos[len(treepos)-1] == 0:

							# Check if tree is completed and then end generation
							if treepos[0] == 0 and len(treepos) == 1:
								# Tree has been completed
								finished = True
							# Important! Reduce current tree position in order to move back up the tree
							del treepos[-1]
							n -= 1
						else:
							# Account for added child
							treepos[len(treepos)-1] = int(treepos[len(treepos)-1]) - 1
							if type(children[i].genome[n]) is Operator:

								subtree[i].append(children[i].genome[n])
								treepos.append(children[i].genome[n].size)
							else:
								subtree[i].append(children[i].genome[n])

					# Remove subtree from genome before crossover
					for z in range(point[i]+1, n+1):
						del children[i].genome[point[i]+1]
				else:
					# Crossover selection is just a node 
					subtree[i].append(branch[i])

			children[0].genome[point[0]] = subtree[1]
			children[1].genome[point[1]] = subtree[0]

			flat = []
			# Flatten genome
			for z in range(0,2):
				for x in range(0,len(children[z].genome)):
					if type(children[z].genome[x]) is list:
						for y in range(0,len(children[z].genome[x])):
							flat.append(children[z].genome[x][y])
					else:
						flat.append(children[z].genome[x])
				children[z].genome = flat
				# print '\nflattended ind, ' ,flat
				flat = []
			
			# Generate children ids and record parents

			children[0].parents = list([parents[0].id, parents[1].id])
			children[1].parents = list([parents[0].id, parents[1].id])

			# print('First child parents are...  ', children[0].parents)
			# print('Second child parents are...  ', children[1].parents)
		
			# Generate new ids
			children[0].id = random.randint(0,2**32)
			children[1].id = random.randint(0,2**32)

			newpop.append(children[0])
			newpop.append(children[1])

	return newpop


def crossover_single(offspring):

	newpop = []
	
	branch = [];children = [];point = []
		
	children.append(offspring[0].copy())
	children.append(offspring[1].copy())	
	# Choose crossover points
	for c in range(0, 2):
		hasop = False
		# Check whether tree has operator
		for d in range(1, len(children[c].genome)):
			if type(children[c].genome[d]) is Operator:
				hasop = True

		if random.random() <= 0.9 and hasop == True:
			# Choose operator
			p = random.randint(1,len(children[c].genome)-1)
			while type(children[c].genome[p]) is not Operator:
				 p = random.randint(1,len(children[c].genome)-1)
		else:
			# Choose leaf node
			p = random.randint(1,len(children[c].genome)-1)
			while type(children[c].genome[p]) is Operator:
				 p = random.randint(1,len(children[c].genome)-1)
		point.append(p)
	
	branch.append(children[0].genome[point[0]])
	branch.append(children[1].genome[point[1]])

	flat = []
	# Flatten genome
	for z in range(0,2):
		for x in range(0,len(children[z].genome)):
			if type(children[z].genome[x]) is list:
				for y in range(0,len(children[z].genome[x])):
					flat.append(children[z].genome[x][y])
			else:
				flat.append(children[z].genome[x])
		children[z].genome = flat
		
		flat = []

	subtree = [[],[]]

	for i in range(0,2):
		# Build the subtree for crossover
		if type(branch[i]) is Operator:

			finished = False
			treepos = []

			# Add new counter for children
			treepos.append(children[i].genome[point[i]].size)
			subtree[i].append(children[i].genome[point[i]])
			n = (point[i]-1) + 1 
			while finished != True:
				n += 1
				# End current operator if max children is reached.
				if treepos[len(treepos)-1] == 0:

					# Check if tree is completed and then end generation
					if treepos[0] == 0 and len(treepos) == 1:
						# Tree has been completed
						finished = True
					# Important! Reduce current tree position in order to move back up the tree
					del treepos[-1]
					n -= 1
				else:
					# Account for added child
					treepos[len(treepos)-1] = int(treepos[len(treepos)-1]) - 1
					if type(children[i].genome[n]) is Operator:

						subtree[i].append(children[i].genome[n])
						treepos.append(children[i].genome[n].size)
					else:
						subtree[i].append(children[i].genome[n])

			# Remove subtree from genome before crossover
			for z in range(point[i]+1, n+1):
				del children[i].genome[point[i]+1]
		else:
			subtree[i].append(branch[i])

	children[0].genome[point[0]] = subtree[1]
	children[1].genome[point[1]] = subtree[0]

	flat = []
	# Flatten genome
	for z in range(0,2):
		for x in range(0,len(children[z].genome)):
			if type(children[z].genome[x]) is list:
				for y in range(0,len(children[z].genome[x])):
					flat.append(children[z].genome[x][y])
			else:
				flat.append(children[z].genome[x])
		children[z].genome = flat
		
		flat = []

	newpop.append(children[0])
	newpop.append(children[1])

	return random.choice([children[0], children[1]])
	
def mutate(offspring, mutrate, growrate, growdepth, blackboard):

	for a in range(0, len(offspring)):
		for b in range(0, len(offspring[a].genome)):

			if random.random() <= mutrate:
				if random.random() <= growrate and b != 0 and b < len(offspring[a].genome) and type(offspring[a].genome[b]) is not Operator:
					# Grow a randomly generated tree
					
					subtree = tg.tree().make_tree(growdepth, blackboard)

					del offspring[a].genome[b]
					offspring[a].genome[b:b] = subtree[:]

				else:
					if type(offspring[a].genome[b]) is Operator:
						offspring[a].genome[b].type = random.choice(blackboard["operators"])

					if type(offspring[a].genome[b]) is Action:
						offspring[a].genome[b].type = random.choice(blackboard["actions"])

					if type(offspring[a].genome[b]) is Param:

						choice = random.choice(['type','param'])

						if choice == 'type':
							# Switch behaviour to another random behaviour
							node = tg.Param()
							node.generate(blackboard)
							offspring[a].genome[b] = node.copy()
						else:
							if offspring[a].genome[b].type == 'random':
								offspring[a].genome[b].param = random.choice(blackboard["randparam"])
							elif offspring[a].genome[b].type == 'rotate_clock' or offspring[a].genome[b].type == 'rotate_anti':
								offspring[a].genome[b].param = random.choice(blackboard["rotparam"])
							else:
								offspring[a].genome[b].param = random.choice(blackboard["dirparam"])


					if type(offspring[a].genome[b]) is Env_control:

						choice = random.choice(['type','pos'])
						if choice == 'type':
							offspring[a].genome[b].type = random.choice(blackboard["envcontrol"])
						if choice == 'pos':
							offspring[a].genome[b].pos[0] = random.choice(blackboard["envcontrol_x"])
							offspring[a].genome[b].pos[1] = random.choice(blackboard["envcontrol_y"])
					
					if type(offspring[a].genome[b]) is Condition:
						
						choice = random.choice(['var','value','op'])
						if choice == 'var':
							# Generate new condition node
							node = tg.Condition()
							node.generate(blackboard)
							offspring[a].genome[b] = node
						if choice == 'op':
							offspring[a].genome[b].op = random.choice(['<','>'])
						if choice == 'value':
							if offspring[a].genome[b].var == 'density':
								offspring[a].genome[b].value = random.choice(blackboard["density"])
							if offspring[a].genome[b].var == 'centery':
								offspring[a].genome[b].value = random.choice(blackboard["dimy"])
							if offspring[a].genome[b].var == 'centerx':
								offspring[a].genome[b].value = random.choice(blackboard["dimx"])
							if offspring[a].genome[b].var == 'agentsy':
								offspring[a].genome[b].value = random.choice(blackboard["dimy"])
							if offspring[a].genome[b].var == 'agentsx':
								offspring[a].genome[b].value = random.choice(blackboard["dimx"])
							if offspring[a].genome[b].var == 'mediany':
								offspring[a].genome[b].value = random.choice(blackboard["dimy"])
							if offspring[a].genome[b].var == 'medianx':
								offspring[a].genome[b].value = random.choice(blackboard["dimx"])
							if offspring[a].genome[b].var == 'coverage':
								offspring[a].genome[b].value = random.choice(blackboard["coverage"])
							if offspring[a].genome[b].var == 'belief':
								offspring[a].genome[b].value = random.choice(blackboard["belief"])

	return offspring


def extinction(population, deathrate, indsize, blackboard):

	# Kill off proportion of population and replace with new individuals
	for n in range(len(population)):

		if random.uniform(0,1) <= deathrate:
			# kill individual and replace with new
			population[n] = tg.individual(tg.tree().make_tree(indsize, blackboard))

	return population





def mutate_single(ind, mutrate, growrate, growdepth, blackboard):

	
	for b in range(0, len(ind.genome)):

		if random.random() <= mutrate:
			if random.random() <= growrate and b != 0 and b < len(ind.genome) and type(ind.genome[b]) is not Operator:
				# Grow a randomly generated tree
				
				subtree = tg.tree().make_tree(growdepth, blackboard)

				del ind.genome[b]
				ind.genome[b:b] = subtree[:]

			else:
				if type(ind.genome[b]) is Operator:
					ind.genome[b].type = random.choice(blackboard["operators"])

				if type(ind.genome[b]) is Action:
					ind.genome[b].type = random.choice(blackboard["actions"])

				if type(ind.genome[b]) is Param:

					choice = random.choice(['type','param'])


					if choice == 'type':
						node = tg.Param()
						node.generate(blackboard)
						ind.genome[b] = node.copy()

					if choice == 'param':
						if ind.genome[b].type == 'random':
							ind.genome[b].param = random.choice(blackboard["randparam"])
						elif ind.genome[b].type == 'rot_anti':
							ind.genome[b].param = random.choice(blackboard["rotparam"])
						elif ind.genome[b].type == 'rot_clock':
							ind.genome[b].param = random.choice(blackboard["rotparam"])
						else:
							ind.genome[b].param = random.choice(blackboard["dirparam"])

				if type(ind.genome[b]) is Env_control:

					choice = random.choice(['type','pos'])
					if choice == 'type':
						ind.genome[b].type = random.choice(blackboard["envcontrol"])
					if choice == 'pos':
						ind.genome[b].pos[0] = random.choice(blackboard["envcontrol_x"])
						ind.genome[b].pos[1] = random.choice(blackboard["envcontrol_y"])
				
				if type(ind.genome[b]) is Condition:
					
					choice = random.choice(['var','value','op'])
					if choice == 'var':
						# Generate new condition node
						node = tg.Condition()
						node.generate(blackboard)
						ind.genome[b] = node.copy()
					if choice == 'op':
						ind.genome[b].op = random.choice(['<','>'])
					if choice == 'value':
						if ind.genome[b].var == 'density':
							ind.genome[b].value = random.choice(blackboard["density"])
						if ind.genome[b].var == 'centery':
							ind.genome[b].value = random.choice(blackboard["centery"])
						if ind.genome[b].var == 'centerx':
							ind.genome[b].value = random.choice(blackboard["centerx"])
						if ind.genome[b].var == 'coverage':
							ind.genome[b].value = random.choice(blackboard["coverage"])

	return ind



def reduction(ind):

	# Function to remove dead parts of a BT

	# Rule 1: If a selector has an action as it's first child. Replace the sub-tree with the action node.
	# Iterate through tree nodes

	DEBUG = False

	n = 0
	treesize = len(ind.genome)

	while n < treesize:

		if DEBUG is True:
			print('\nAt point ', n)

			if type(ind.genome[n]) is Operator:
				print('Operator ', ind.genome[n].type)
			if type(ind.genome[n]) is Condition:
				print('Condition ', ind.genome[n].var)
			if type(ind.genome[n]) is Action:
				print('Action ', ind.genome[n].type)
			if type(ind.genome[n]) is Param:
				print('Param ', ind.genome[n].type)
			if type(ind.genome[n]) is Env_control:
				print('Env ', ind.genome[n].type)

		# check if node is a selector
		if type(ind.genome[n]) is Operator and ind.genome[n].type == 'sel':

			# Is the first child an action node?
			if type(ind.genome[n+1]) is not Condition and type(ind.genome[n+1]) is not Operator:

				if DEBUG is True:
					print('\n\nFound dead tree!')
				# Then the subtree starting with the selector needs to be replaced.

				# make a copy of the action to replace the sub-tree
				replace_node = ind.genome[n+1].copy()

				# Check the numder of children for the selector
				size = ind.genome[n].size
				''' 
				Remove subtree from genome before replacement.
				If the selector contains sub-trees they also need to be removed.
				'''

				# Track depth within tree
				treepos = []
				treepos.append(ind.genome[n].size)

				finished = False
				z = n 

				while finished is False:
					z += 1

					if DEBUG is True:
						# print('Z position ', z)
						# print('Treepos: ', treepos)
						# print(type(ind.genome[z]))

						if type(ind.genome[z]) is Operator:
							print('Operator ', ind.genome[z].type)
						if type(ind.genome[z]) is Condition:
							print('Condition ', ind.genome[z].var)
						if type(ind.genome[z]) is Action:
							print('Action ', ind.genome[z].type)
						if type(ind.genome[z]) is Param:
							print('Param ', ind.genome[z].type)
						if type(ind.genome[z]) is Env_control:
							print('Env ', ind.genome[z].type)


					if treepos[len(treepos)-1] == 0:

						# Check if finished removal
						if treepos[0] == 0 and len(treepos) == 1:
							# Tree has been completed
							finished = True
						# Important! Reduce current tree position in order to move back up the tree
						del treepos[-1]
						z -= 1

					else:
						# Account for deleted child
						treepos[len(treepos)-1] = int(treepos[len(treepos)-1]) - 1
						
						if type(ind.genome[z]) is Operator:

							treepos.append(ind.genome[z].size)
							
				endpoint = z

				# Remove subtree from genome 
				for z in range(n+1, endpoint+1):

					if DEBUG is True:
						print('DELETING')
						if type(ind.genome[n+1]) is Operator:
							print('Operator ', ind.genome[n+1].type)
						if type(ind.genome[n+1]) is Condition:
							print('Condition ', ind.genome[n+1].var)
						if type(ind.genome[n+1]) is Action:
							print('Action ', ind.genome[n+1].type)
						if type(ind.genome[n+1]) is Param:
							print('Param ', ind.genome[n+1].type)
						if type(ind.genome[n+1]) is Env_control:
							print('Env ', ind.genome[n+1].type)
					del ind.genome[n+1]

				# Replace operator node with action node
				ind.genome[n] = replace_node
				
				if DEBUG is True:
					print('Replaced node ', type(ind.genome[n]))

		# Update treesize due to sub-tree removal
		treesize = len(ind.genome)
		# Go to next position in tree
		n += 1		



	# Rule 2: An action (A) followed by an action (B), overwrites action (A). 		 

	n = 0
	treesize = len(ind.genome)

	while n < treesize-1:

		if DEBUG is True:
			print('\nAt point ', n)

			if type(ind.genome[n]) is Operator:
				print('Operator ', ind.genome[n].type)
			if type(ind.genome[n]) is Condition:
				print('Condition ', ind.genome[n].var)
			if type(ind.genome[n]) is Action:
				print('Action ', ind.genome[n].type)
			if type(ind.genome[n]) is Param:
				print('Param ', ind.genome[n].type)
			if type(ind.genome[n]) is Env_control:
				print('Env ', ind.genome[n].type)


		# Track the current parent in the tree
		
		if type(ind.genome[n]) is Operator:
			parent = n	

		# check if node is an Action
		if type(ind.genome[n]) is Action or type(ind.genome[n]) is Param:

			# Is the next node also an Action?
			if type(ind.genome[n+1]) is Action or type(ind.genome[n+1]) is Param:


				if ind.genome[parent].size > 2:
					# Then we need to replace the first action with the second.
					# make a copy of the action to replace the sub-tree

					# First account for the child reduction of the parent
					ind.genome[parent].size -= 1

					replace_node = ind.genome[n+1].copy()

					# Replace operator node with action node
					ind.genome[n] = replace_node

					# Delete second action
					del ind.genome[n+1]

					if DEBUG is True:
						print('Replaced node ', type(ind.genome[n]))

		# Update treesize due to sub-tree removal
		treesize = len(ind.genome)
		# Go to next position in tree
		n += 1		

	# Return the reduced tree
	return ind




def load_traveller(file, num_travellers):

	# Read the travellers file and load the best individual.
	with open(file, 'rb') as input:
		travellers = pickle.load(input)


	# load a random traveller
	new_travellers = []
	for i in range(num_travellers):
		pick = np.random.randint(0, len(travellers))

		new_travellers.append(travellers[pick].copy())
	
	return new_travellers


def save_traveller(pop, tot_travellers, num_travellers, file):

	# Save the best solution from pop to the travellers file.

	with open(file, 'rb') as input:
		travellers = pickle.load(input)

	# Open travellers and add best solutions from pop

	travellers = travellers + pop[:num_travellers]
	travellers.sort(key=lambda x: x.fitness, reverse=True)

	# Sort solutions and only keep the best.
	travellers = [ind.copy() for ind in travellers[0:tot_travellers]]

	# Write to file new travellers list.
	with open(file, 'wb') as output:
		pickle.dump(travellers, output, pickle.HIGHEST_PROTOCOL)





def hallfame(pop, hall, hallsize):

	hall = hall + pop
	hall.sort(key=lambda x: x.fitness, reverse=True)
	hall = [ind.copy() for ind in hall[0:hallsize]]

	return hall

def save_gen(gens, file):
	# Save population
	with open(file, 'wb') as output:
		pickle.dump(gens, output, pickle.HIGHEST_PROTOCOL)

def checkpoint_gen(pop, file):
	# Save population
	with open(file, 'wb') as output:
		pickle.dump(pop, output, pickle.HIGHEST_PROTOCOL)


def tournament(pop, tournsize, selectionNum):
	sel = []
	newpop = []
	topfit = -9999; fit = 0; best = 0

	for n in range(0, selectionNum):
		# Reset best fittness and choice list!!!!
		topfit = -9999
		choice = list(range(0,len(pop)))
		for i in range(0, tournsize):
			# make a selection of the availible individuals in the choice list.
			sel = (random.randint(0,(len(choice)-1)))
			fit = pop[choice[sel]].fitness
			if fit > topfit:
				# Track individual with best fitness
				best = choice[sel]
				topfit = fit
			# remove selected indivudal from list.
			choice.remove(choice[sel])
		newpop.append(pop[best])

	return newpop

def neighbour_selection(pop):
	sel = []
	newpop = []
	topfit = -9999; fit = 0; best = 0

	# Pick best two inds from neighbourhood

	for n in range(0, 2):
		# Reset best fittness and choice list!!!!
		topfit = -9999
		choice = list(range(0,len(pop)))
		for i in range(0, len(pop)):
			# make a selection of the availible individuals in the choice list.
			sel = (random.randint(0,(len(choice)-1)))
			fit = pop[choice[sel]].fitness
			if fit > topfit:
				# Track individual with best fitness
				best = choice[sel]
				topfit = fit
			# remove selected indivudal from list.
			choice.remove(choice[sel])
		newpop.append(pop[best])

	return newpop

def log(pop, hall, logpop, logfit, logavg, logmax, g):

	f = []; tot = 0; maxfit = 0

	for i in range(0, len(pop)):
		f.append(pop[i].fitness)
		tot += pop[i].fitness
		
		if pop[i].fitness > maxfit:
			maxfit = pop[i].fitness
	
	avg = tot/len(pop)
	logfit.append(f)
	logmax.append(maxfit)
	logavg.append(avg)
	logpop.append(pop)

	print('\n' + 80*'-')
	print('\nFINISHED GENERATION')
	
	print('\nAverage Fitness: %s' % logavg)
	print('\nMaximum Fitness: %s' % logmax)
	print(80*'-')



def gen_analysis(pop, stats, filename):

	treesizes = np.zeros(len(pop))
	fits = np.zeros(len(pop))

	for n in range(len(pop)):

		treesizes[n] = len(pop[n].genome)
		fits[n] = pop[n].fitness

	avgsize = np.mean(treesizes)
	stdsize = np.std(treesizes)
	avgfit = np.mean(fits)
	stdfit = np.std(fits)
	maxfit = np.max(fits)

	stats["avgsize"].append(avgsize)
	stats["stdsize"].append(stdsize)
	stats["meanfit"].append(avgfit)
	stats["maxfit"].append(maxfit)
	stats["stdfit"].append(stdfit)

	with open(filename + '_stats', 'wb') as output:
		pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)

	return stats





def log_settings(seed,swarm,blackboard,popsize,indsize,tournsize,elitesize,newind,mutrate,
	growrate,growdepth,op_prob,treecost,extinction_rate,death_rate,ngen,targets,state,timesteps,filename, env_name, door_states = 'none'):

	file = open(filename, 'w+')
	file.write('\nRun settings: --->')
	file.write('\n\nRandom seed: %d' % seed)
	file.write('\n\nSwarm size: %d' % swarm.size)
	file.write('\n\nSwarm agent speed: %f' % swarm.speed)
	file.write('\n\nSwarm map: %s' % swarm.map)
	file.write('\nTarget setup: %s' % state)
	file.write('\nTarget detection radius: %d' % targets.radius)
	file.write('\n\nBlackboard: %s' % blackboard)
	file.write('\n\nNumber of generations: %d' % ngen)
	file.write('\n\nTest length: %d' % timesteps)
	file.write('\nPopulation size: %d' % popsize)
	file.write('\nBehaviour tree initial depth: %d' % indsize)
	file.write('\nTournament size: %d' % tournsize)
	file.write('\nElite size: %d' % elitesize)
	file.write('\nNew individuals: %d' % newind)
	file.write('\nMutation rate: %f' % mutrate)
	file.write('\nCrossover of sub-tree prob: %f' % op_prob)
	file.write('\nSubtree growth rate: %f' % growrate)
	file.write('\nSubtree growth depth: %d' % growdepth)
	file.write('\nTree size cost: %f' % treecost)
	file.write('\n\nExtinction probability: %f' % extinction_rate)
	file.write('\nProbability of death: %f' % death_rate)

	file.write('\n\nEnvironment name: %s' % env_name)
	file.write('\nEnvironment door states: '+str(door_states))
	file.close()


def log_settings_cellular(seed,netdim,neighbourhood,swarm,popsize,indsize,tournsize,elitesize,newind,mutrate,growrate,growdepth,ngen,targets,state,timesteps,filename):


	file = open(filename, 'w+')
	file.write('\nRun settings: --->')
	file.write('\nEvolved using a cellular evolutionary algorithm')
	file.write('\nPopulation network generated with dimension: %d' % netdim)
	file.write('\nPopulation network generated with neighbourhood: %d' % neighbourhood)

	file.write('\n\nRandom seed: %d' % seed)
	file.write('\n\nSwarm size: %d' % swarm.size)
	file.write('\n\nSwarm agent speed: %f' % swarm.speed)
	file.write('\n\nSwarm map: %s' % swarm.map)
	
	file.write('\nTarget setup: %s' % state)
	file.write('\nTarget detection radius: %d' % targets.radius)
	file.write('\n\nNumber of generations: %d' % ngen)
	file.write('\n\nTest length: %d' % timesteps)
	file.write('\nPopulation size: %d' % popsize)
	file.write('\nBehaviour tree initial depth: %d' % indsize)
	file.write('\nTournament size: %d' % tournsize)
	file.write('\nElite size: %d' % elitesize)
	file.write('\nNew individuals: %d' % newind)
	file.write('\nMutation rate: %f' % mutrate)
	file.write('\nSubtree growth rate: %f' % growrate)
	file.write('\nSubtree growth depth: %d' % growdepth)
	file.close()


