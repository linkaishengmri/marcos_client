#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import copy, os, socket, sys, time, warnings
import multiprocessing as mp

from device import Device
import server_comms as sc
import test_base as tb
from mimo_devices import mimo_dev_run

import local_config as lc

def reset_grad():
    
    
    params_shared = {
        'halt_and_reset': True, 
        'flush_old_rx': True, 
        'init_gpa': True,
        # "lo_freq": 20,
        # "rx_t": 0.0326,
        # "print_infos": True,
        # "assert_errors": True,
        # "halt_and_reset": False,
        # "fix_cic_scale": True,
        # "set_cic_shift": False,  # needs to be true for open-source cores
        # "flush_old_rx": False,
    }
    
    dev_m = Device(
        ip_address=lc.ip_address,
        port=lc.port,
        mimo_master=True,
        trig_output_time=1e5,
        **(params_shared)
    )
    dev_s = Device(
        ip_address=lc.ip_address_slave,
        port=lc.port_slave,
        trig_timeout=10,
        **(params_shared)
    )
    dev_m.add_flodict({ 
                        'tx0':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vy':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz2':[np.array([100, 100000]), np.array([.01,0])],
                        'rx0_en':[np.array([100, 100000]), np.array([1,0])],
                        })
    dev_s.add_flodict({ 
                        'tx0':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vy':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz2':[np.array([100, 100000]), np.array([.01,0])],
                        'rx0_en':[np.array([100, 100000]), np.array([1,0])],
                        })
    
    rev1 = mimo_dev_run((dev_m, 0))
    print(rev1)
    rev2 = mimo_dev_run((dev_s, 0))
    print(rev2)
    dev_m.close_server(only_if_sim=True)
    dev_s.__del__()   

if __name__ == '__main__':
    reset_grad()
