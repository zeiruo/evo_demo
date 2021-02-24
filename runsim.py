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


def flocking(swarm, repel, attract, comm_range, noise):

	R = repel; r = 3.5; A = attract; a = 5.5

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Determine headings
	nearest = mag <= comm_range

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
	swarm.agents = bsim.continuous_boundary(swarm.agents, swarm.map)
	


swarmsize = 300
swarm = bsim.swarm()
swarm.size = swarmsize
swarm.speed = 0.5

env = bsim.map()
env.empty()
env.gen()
swarm.map = env

swarm.gen_agents_uniform(env)

timesteps = 600

# First set up the figure, the axis, and the plot element we want to animate
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(16,8), dpi=80, facecolor='w', edgecolor='k')
#plt.close()

dim = 40
ax1.set_xlim((-dim, dim))
ax1.set_ylim((-dim, dim))

ax2.set_xlim((0, timesteps))
ax2.set_ylim((0, 1))

# Set how data is plotted within animation loop
global line, line1
# Agent plotting 
line, = ax1.plot([], [], 'rh', markersize = 6, markeredgecolor="black", alpha = 0.9)
# shadow plotting
line1, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.2)
line2, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.2)
line3, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.2)

box_line, = ax2.plot([],[], 'r-', markersize = 5)

fsize = 12

time_text = ax1.text(-20, 26, '', fontsize = fsize)
box_text = ax1.text(3, 26, '', color = 'red', fontsize = fsize)

line.set_data([], [])
line1.set_data([], [])

def init():
    line.set_data([], [])
    line1.set_data([], [])
    old_positions = np.zeros((swarm.size,2))
    return (line, line1, box_line,)


ax2.set_yticks(np.arange(0, 1, 0.1))
ax2.grid()

# plot the walls
[ax1.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
    [swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

# Declare agent motion noise
noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))
score = 0

# Set the swarms behaviour
swarm.behaviour = 'flocking'
swarm.param = 0.07

box_data = []
time_data = []


def animate(i):

    #swarm.iterate(noise[i-1])
    flocking(swarm, repel=30, attract=10, comm_range=3, noise=noise[i-1])
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

    return ( line1, line2, line3, box_line, time_text, line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=timesteps, interval=50, blit=True)

#anim.save('try_animation.mp4', fps=20, dpi=120)
# Note: below is the part which makes it work on Colab
#rc('animation', html='jshtml')
anim
plt.show()