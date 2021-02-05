#!/usr/bin/env python
import math
import numpy
import random
import numpy as np
import pickle
import subprocess
import py_trees
import itertools
import re
import os
import sys

from simulation.asim import*
from inspect import isclass


class beh_param(py_trees.behaviour.Behaviour):

	def __init__(self, name="beh_param"):
		"""
		Default construction.
		"""
		super(beh_param, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.vector = [0,0]
		self.swarm = []
		self.param = 0
		self.command = ''


	def setup(self, unused_timeout=15):
		"""
		No delayed initialisation required for this example.
		"""
		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return py_trees.common.Status.INVALID

	def initialise(self):
		"""
		Reset a counter variable.
		"""

	def update(self):
		"""
		Increment the counter and decide upon a new common.Status result for the behaviour.
		"""
		self.swarm.behaviour = self.command
		self.swarm.param = self.param
		return py_trees.common.Status.SUCCESS

	def terminate(self, new_status):
		"""
		Nothing to clean up in this example.
		"""
		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
		return py_trees.common.Status.INVALID

class env_control(py_trees.behaviour.Behaviour):

	def __init__(self, name="env_control"):
		"""
		Default construction.
		"""
		super(env_control, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.swarm = []
		self.beacon = []
		self.set = False

	def setup(self, unused_timeout=15):
		"""
		No delayed initialisation required for this example.
		"""
		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return py_trees.common.Status.INVALID

	def initialise(self):
		"""
		Reset a counter variable.
		"""

	def update(self):
		"""
		Increment the counter and decide upon a new common.Status result for the behaviour.
		"""
		if self.set == False:
			if self.beacon.type  == 'attract':
				if self.swarm.beacon_att.size == 0:
					self.swarm.beacon_att = np.array([[self.beacon.pos[0], self.beacon.pos[1]]])
				else:	
					self.swarm.beacon_att = np.append(self.swarm.beacon_att, np.array([[self.beacon.pos[0], self.beacon.pos[1]]]), axis=0)
			
			if self.beacon.type  == 'repel':
				if self.swarm.beacon_rep.size == 0:
					self.swarm.beacon_rep = np.array([[self.beacon.pos[0], self.beacon.pos[1]]])
				else:	
					self.swarm.beacon_rep = np.append(self.swarm.beacon_rep, np.array([[self.beacon.pos[0], self.beacon.pos[1]]]), axis=0)
			self.set = True
		return py_trees.common.Status.SUCCESS

	def terminate(self, new_status):
		"""
		Nothing to clean up in this example.
		"""
		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class behaviour(py_trees.behaviour.Behaviour):

	def __init__(self, name="beh_param"):
		"""
		Default construction.
		"""
		super(behaviour, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.vector = [0,0]
		self.swarm = []
		self.param = 0
		self.command = ''

	def setup(self, unused_timeout=15):
		"""
		No delayed initialisation required for this example.
		"""
		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return py_trees.common.Status.INVALID

	def initialise(self):
		"""
		Reset a counter variable.
		"""

	def update(self):
		"""
		Increment the counter and decide upon a new common.Status result for the behaviour.
		"""
		self.swarm.behaviour = self.command
		self.swarm.param = 10
		# if self.swarm.behaviour == 'aggregate':
		# 	self.swarm.param = 35
		# if self.swarm.behaviour == 'random':
		# 	self.swarm.param = 0.01
		# if self.swarm.behaviour == 'rot_anti' or 'rot_clock':
		# 	self.swarm.param = 0.03
		# else:
		# 	self.swarm.param = 3

		
		return py_trees.common.Status.SUCCESS

	def terminate(self, new_status):
		"""
		Nothing to clean up in this example.
		"""
		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))




class LessY(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessY"):

		super(LessY, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		#print 'variable: ', self.var
		self.explore = False
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return py_trees.common.Status.INVALID

	def initialise(self):

		return True

	def update(self):
		
		self.var = self.swarm.centermass[1]
		if self.var <= self.const and self.explore == False:
			self.explore = True
			
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class GreaterY(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterY"):

		super(GreaterY, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		#print 'variable: ', self.var
		self.explore = False
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):
		
		self.var = self.swarm.centermass[1]

		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class LessX(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessX"):

		super(LessX, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		#print 'variable: ', self.var
		self.explore = False
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):
		
		self.var = self.swarm.centermass[0]
		if self.var <= self.const and self.explore == False:
			self.explore = True
			
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class GreaterX(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterX"):

		super(GreaterX, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		#print 'variable: ', self.var
		self.explore = False
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):
		
		self.var = self.swarm.centermass[0]
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
		return py_trees.common.Status.INVALID

class LessDense(py_trees.behaviour.Behaviour, swarm):

	def __init__(self, name="LessDense"):

		super(LessDense, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0

		#print 'variable: ', self.var
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):
		
		self.var = self.swarm.spread
		#print 'dense ', density
		if self.var <= self.const:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class GreaterDense(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterDense"):

		super(GreaterDense, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		#print 'variable: ', self.var
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.spread
		#print 'dense ', density
		if self.var >= self.const:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterCov(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterCov"):

		super(GreaterCov, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.targets = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.targets.coverage
		#print 'dense ', density
		if self.var >= self.const:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class LessCov(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessCov"):

		super(LessCov, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.targets = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.targets.coverage
		#print 'dense ', density
		if self.var <= self.const:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))	


class GreaterAgentx(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterAgentx"):

		super(GreaterAgentx, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.agents
		
		# Check if any agents are above threshold position

		if np.sum(self.var.T[0] >= self.const) >= 1:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class LessAgentx(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessAgentx"):

		super(LessAgentx, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.agents
		
		# Check if any agents are above threshold position

		if np.sum(self.var.T[0] <= self.const) >= 1:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterAgenty(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterAgenty"):

		super(GreaterAgenty, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.agents
		
		# Check if any agents are above threshold position

		if np.sum(self.var.T[1] >= self.const) >= 1:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))		

class LessAgenty(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessAgenty"):

		super(LessAgenty, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.agents
		
		# Check if any agents are above threshold position

		if np.sum(self.var.T[1] <= self.const) >= 1:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterMedianx(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterMedianx"):

		super(GreaterMedianx, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.median[0]
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class LessMedianx(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessMedianx"):

		super(LessMedianx, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.median[0]
		#print 'y ', centery
		if self.var <= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterMediany(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterMediany"):

		super(GreaterMediany, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.median[1]
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))		

class LessMediany(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessMediany"):

		super(LessMediany, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.median[1]
		
		#print 'y ', centery
		if self.var <= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))



class LessUpperX(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessUpperX"):

		
		super(LessUpperX, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.upper[0]
		
		#print 'y ', centery
		if self.var <= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class GreaterUpperX(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterUpperX"):

		
		super(GreaterUpperX, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.upper[0]
		
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class LessUpperY(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessUpperY"):

		
		super(LessUpperY, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.upper[1]
		
		#print 'y ', centery
		if self.var <= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterUpperY(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterUpperY"):

		
		super(GreaterUpperY, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.upper[1]
		
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class LessLowerX(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessLowerX"):

		
		super(LessLowerX, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.lower[0]
		
		#print 'y ', centery
		if self.var <= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterLowerX(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterLowerX"):

		
		super(GreaterLowerX, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.lower[0]
		
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

class LessLowerY(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessLowerY"):

		
		super(LessLowerY, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.lower[1]
		
		#print 'y ', centery
		if self.var <= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterLowerY(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterLowerY"):

		
		super(GreaterLowerY, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.lower[1]
		
		#print 'y ', centery
		if self.var >= self.const and self.explore == False:
			self.explore = True
			return py_trees.common.Status.SUCCESS
		if self.explore == True:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.RUNNING

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class GreaterBelief(py_trees.behaviour.Behaviour):

	def __init__(self, name="GreaterBelief"):

		
		super(GreaterBelief, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.belief

		if self.var <= self.const:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class LessBelief(py_trees.behaviour.Behaviour):

	def __init__(self, name="LessBelief"):

		
		super(LessBelief, self).__init__(name)
		self.logger.debug("%s.__init__()" % (self.__class__.__name__))
		self.var = 0
		self.mode = ''
		self.const = 0
		self.swarm = []
		self.explore = False

	def setup(self, unused_timeout=15):

		self.logger.debug("%s.setup()->connections to an external process" % (self.__class__.__name__))
		return True

	def initialise(self):

		return True

	def update(self):

		self.var = self.swarm.belief

		if self.var <= self.const:
			return py_trees.common.Status.SUCCESS
		else:
			return py_trees.common.Status.FAILURE

	def terminate(self, new_status):

		self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))





