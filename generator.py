from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer

class Generator(object):

    def __init__(self, arch, is_training=False):
        self.arch = arch
 	self._sanity_check()
        self.is_training = is_training

     def _sanity_check(self):
        for net in ['encoder', 'generator']:
	    assert len(self.arch[net]['output']) == len(self.arch[net]['kernel']) == len(self.arch[net]['stride'])   



    

