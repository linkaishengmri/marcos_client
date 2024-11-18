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

    def __init__(self, ips, ports, trig_output_time=0, trig_latency=6.08,
                 trig_timeout=136533, master_run_delay=0, custom_args=None, **kwargs):
        """ips: list of device IPs, master first

        ports: list of device ports, master first

        trig_output_time: usec, when to trigger the slaves after beginning the
        sequence on the master

        trig_latency: usec, how long the slaves take from being triggered by the
        master to beginning their sequences (plus any additional I/O or cable
        latencies)

        trig_timeout: usec, how long should the slaves wait for a trigger until
        they run their preprogrammed sequences anyway. Same behaviour and
        limitations as for the Device class. Negative values = infinite timeout
        so only use this when the system is debugged.

        master_run_delay: sec, how long to wait for the slaves to start running
        before starting the master compilation/programming/execution -- the
        master must begin after the slaves are awaiting a trigger, otherwise
        sync will not be maintained. Positive values will delay the master's
        run() call, negative values will delay the slaves. [TODO: Also accepts a
        per-device list.]

        extra_args: list of dictionaries of extra arguments to each Device object, master first

        All remaining arguments supported by the Device class are also
        supported, and will be passed down to each Device.

        """

        devN = len(ips)

        assert (len(ips) == len(ports)) and (len(ports) == len(extra_args)), f"Lengths of ips/ports/extra_args ({len(ips)}/{len(ports)}/{len(extra_args)}) unequal"

        if extra_args is None:
            device_args = [kwargs] * devN
        else:
            device_args = list(ea | kwargs for ea in extra_args)

        master_delay = 0
        slave_delay = 0
        if master_run_delay > 0:
            master_rd = master_run_delay
        else:
            slave_rd = -master_run_delay

        self._devs = []
        self._run_delays = []

        for k, (ip, port, devargs) in enumerate(zip(ips, ports, device_args)):
            if k == 0:
                # TODO cannot handle the case where the MIMO system is externally triggered
                devargs['trig_timeout'] = 0
                self._run_delays.append(master_rd)
            else:
                devargs['trig_timeout'] = trig_timeout
                self._run_delays.append(slave_rd)

            dev = Device(
                ip_address=ip, port=port, **devargs)

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
