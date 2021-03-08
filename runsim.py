import random
import sys
import numpy as np
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

from matplotlib import animation, rc, rcParams
rcParams['animation.embed_limit'] = 2**128
#from IPython.display import HTML

'''
The next section will simulate and plot of the best individual
produced at the end of the evolution.
'''


def flocking(swarm, repel, attract, comm_range, align, noise):

	argname = ['repel', 'attract', 'align']
	args = [repel, attract, align]
	for n in range(len(args)):
		if args[n] > 1 or args[n] < 0:
			raise ValueError("Value %s must be within the range of 0 to 1." % (argname[n]))

	R = repel; r = 3; A = attract; a = 3

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Determine headings
	nearest = mag <= comm_range

	# n x n matrix of headings of agents which are adjacent
	neighbour_headings = swarm.headings*nearest

	# Sum headings for each agent
	neighbour_headings_tot = np.sum(neighbour_headings, axis = 1)

	# average headings with neighbours
	new_headings = neighbour_headings_tot/(np.sum(nearest, axis = 1))

	# Determine the difference between current heading and neighbour avg
	heading_diff = swarm.headings - new_headings

	# Adjust heading to neighbours. Degree of alignment determined by align param
	swarm.headings -= (align*heading_diff) + 0.01*np.random.randint(-10,11, swarm.size)

	# Calculate new heading vector
	strength = 10
	gx = strength*np.cos(swarm.headings)
	gy = strength*np.sin(swarm.headings)
	#G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	G = -np.stack((gx, gy), axis = 1)
	

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#Avoid = fieldmap_avoidance(swarm)
	
	repel = R*r*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)	
	repel = np.sum(repel, axis = 0).T

	attract = A*a*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)	
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
	swarm.agents = bsim.continuous_boundary(swarm.agents, swarm.map)
	

# First set up the figure, the axis, and the plot element we want to animate
fig, ax1 = plt.subplots( figsize=(10,10), dpi=80, facecolor='w', edgecolor='k')
#plt.close()

dim = 40
ax1.set_xlim((-dim, dim))
ax1.set_ylim((-dim, dim))

# Set how data is plotted within animation loop
global line, line1
# Agent plotting 
line, = ax1.plot([], [], 'rh', markersize = 6, markeredgecolor="black", alpha = 0.9)
# shadow plotting
line1, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.2)
line2, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.2)
line3, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.2)

fsize = 12

time_text = ax1.text(-20, 26, '', fontsize = fsize)
box_text = ax1.text(3, 26, '', color = 'red', fontsize = fsize)

line.set_data([], [])
line1.set_data([], [])

def init():
    line.set_data([], [])
    line1.set_data([], [])
    return (line, line1,)

swarmsize = 100
swarm = bsim.swarm()
swarm.size = swarmsize
swarm.speed = 0.8

env = bsim.map()
env.empty()
env.gen()
swarm.map = env

swarm.gen_agents_uniform(env)

timesteps = 600

# Declare agent motion noise
noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

# Set the swarms behaviour
swarm.behaviour = 'random'
swarm.param = 0.01

[ax1.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

time_data = []

def animate(i):

    flocking(swarm, repel=0, attract=1, comm_range=3, align = 0.85, noise=noise[i-1])
    swarm.get_state()
    time_data.append(i)
    
    x = swarm.agents.T[0]
    y = swarm.agents.T[1]

    time_text.set_text('Time: (%d/%d)' % (i, timesteps))
  
    
    line1.set_data(swarm.shadows[1].T[0], swarm.shadows[1].T[1])
    line2.set_data(swarm.shadows[2].T[0], swarm.shadows[2].T[1])
    line3.set_data(swarm.shadows[3].T[0], swarm.shadows[3].T[1])
    line.set_data(x, y)
    old_positions = swarm.agents

    return (line1, line2, line3, time_text, line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=timesteps, interval=25, blit=True)

# anim.save('sim_animation.mp4', fps=60, dpi=200)
# Note: below is the part which makes it work on Colab
#rc('animation', html='jshtml')
anim
plt.show()