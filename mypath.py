"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

data_path = '/data/xinyut/datasets/'
ckpt_path = '/data/gpfs/xinyut/ckpts/labeldptest'

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar10', 'cifar20', 'cifar100', 'cinic10'}
        assert(database in db_names)

        return data_path+database

    def ckpt_root_dir():
        return ckpt_path