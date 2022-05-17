import os
import argparse
from argparse import ArgumentParser
import yaml
import numpy as np

class Parser(ArgumentParser):
    
    def __init__(self):
        super().__init__()
        self.command, self.config = self.configure()
               
    def configure(self):
        stream = open("config.yaml", 'r')
        dictionary = yaml.safe_load(stream)
        command = ['python', 'dirichlet.py']
        for item in list(dictionary.keys()):
            i_default = dictionary[item]['default']
            i_type = dictionary[item]['type']
            if i_type == 'bool':
                self.add_argument("--" + item, default = i_default , action='store_true')
            else:
                self.add_argument("--" + item, default = i_default , type = type(i_default))
                command.append("--" + str(item) + ' ' + str(i_default))
            
           
       
        return command, self.parse_args()