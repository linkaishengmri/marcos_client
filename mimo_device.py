#!/usr/bin/env python3
#
# Utility class to operate MIMO systems

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt

import local_config as lc
import grad_board as gb
import server_comms as sc
import marcompile as mc

from device import Device

def mimo_dev_run(devt):
    """Allows the parallelisation of Device run() calls, which is essential
    since the slaves will block otherwise."""
    dev, delay = devt
    time.sleep(delay)
    rxd, msgs = dev.run()
    return rxd, msgs


class MimoDevice:
    """Manages multiple Devices, collates/distributes settings between them and
    handles messages and warnings."""

    def __init__(self, ips, ports, shared_args, individual_args=None,
                 master_offset_time=0, master_run_delay=1, trig_wait_time=16e6):
        """ips: list of device IPs, master first

        ports: list of device ports, master first

        shared_args: dict of shared arguments

        individual_args: list of dicts with individual device arguments, master first

        master_offset_time: usec, offset to apply to all events on the master, to
        synchronise them to the slaves (since the master produces a trigger and
        the slaves await it, their events will be slightly delayed without a
        slight offset)

        master_run_delay: sec, how long to wait for the slaves to start running before
        starting the master programming/execution -- the master must begin after
        the slaves are awaiting a trigger, otherwise sync will not be maintained

        trig_wait_time: how long should the slaves wait for a trigger until they
        run their preprogrammed sequences anyway. Same behaviour as for the
        Device class.

        """

        self._devs = []
        self._mot = master_offset_time
        self._mrd = master_run_delay

        for k, (ip, port, args) in enumerate(zip(ips, ports, individual_args)):
            if k == 0:
                # TODO cannot handle the case where the MIMO system is externally triggered
                twt = 0
            else:
                twt = trig_wait_time

            dev = Device(
                ip_address=ip, port=port, trig_wait_time=twt, **(shared_args | args))

            self._devs.append(dev)

    def get_device(self, k):
        return self._devs[k]

    def run(self):
        """ Runs the Devices in parallel, collates their results and settings """
        delays = np.zeros_like(self._devs)
        delays[0] = self._mrd
        with mp.Pool(len(self._devs)) as p:
            res = p.map(mimo_dev_run, zip(self._devs, delays))

        return res
