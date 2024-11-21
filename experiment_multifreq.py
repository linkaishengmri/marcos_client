#!/usr/bin/env python3
#
# Basic toolbox for server operations; wraps up a lot of stuff to avoid the need for hardcoding on the user's side.

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
import grad_board as gb
import server_comms as sc
import marcompile as fc

import pdb



from experiment import Experiment

class ExperimentMultiFreq(Experiment):
    """
    Enhanced Experiment class to support multi-frequency configurations
    during a sequence.

    In addition to all parameters inherited from the base Experiment class,
    this class supports dynamic updates to local oscillator (LO) frequencies
    and reset signals (`lo*_freq`, `lo*_rst`) at specific time points in
    a sequence.

    This is especially useful for multi-slice or other advanced acquisition
    techniques that require frequency adjustments mid-sequence.
    """

    def __init__(self, *args, UseMultiFreq=False, **kwargs):
        """
        Initializes the ExperimentMultiFreq class.
        Inherits all initialization parameters from the base Experiment class.
        """
        super().__init__(*args, **kwargs)
        # Add attributes specific to multi-frequency support if needed
        self.UseMultiFreq = UseMultiFreq  # Enable/disable multi-frequency support

    
    def flo2int(self, seq_dict):
        """Convert a floating-point sequence dictionary to an integer binary
        dictionary with lo*_freq, lo*_rst params added"""

        intdict = {}

        ## Various functions to handle the conversion
        def times_us(farr):
            """ farr: float array, times in us units; [0, inf) """
            return np.round(fpga_clk_freq_MHz * farr).astype(np.int64) # negative values will get rejected at a later stage

        def tx_real(farr):
            """ farr: float array, [-1, 1] """
            return np.round(32767 * farr).astype(np.uint16)

        def tx_complex(times, farr, tolerance=2e-6):
            """times: float time array, farr: complex float array, [-1-1j, 1+1j]
            tolerance: minimum difference two values need to be considered binary-unique (2e-6 corresponds to ~19 bits)
            -- returns a tuple with repeated elements removed"""
            idata, qdata = farr.real, farr.imag
            unique = lambda k: np.concatenate([[True], np.abs(np.diff(k)) > tolerance])
            # DEBUGGING: use the below lambda instead to avoid stripping repeated values
            # unique = lambda k: np.ones_like(k, dtype=bool)
            idata_u, qdata_u = unique(idata), unique(qdata)
            tbins = ( times_us(times[idata_u] + self._initial_wait), times_us(times[qdata_u] + self._initial_wait) )
            txbins = ( tx_real(idata[idata_u]), tx_real(qdata[qdata_u]) )
            return tbins, txbins

        def lo_real(farr):
            dds_phase_steps = np.round(2**31 / fpga_clk_freq_MHz * np.array(farr)).astype(np.uint32)
            return dds_phase_steps
        
        for key, (times, vals) in seq_dict.items():
            # each possible dictionary entry returns a tuple (even if one element) for the binary dictionary to send to marcompile
            tbin = times_us(times + self._initial_wait),
            if key in ['tx0_i', 'tx0_q', 'tx1_i', 'tx1_q']:
                valbin = tx_real(vals),
                keybin = key,
            elif key in ['tx0', 'tx1']:
                tbin, valbin = tx_complex(times, vals)
                keybin = key + '_i', key + '_q'
            elif key in ['grad_vx', 'grad_vy', 'grad_vz', 'grad_vz2',
                         'fhdo_vx', 'fhdo_vy', 'fhdo_vz', 'fhdo_vz2',
                         'ocra1_vx', 'ocra1_vy', 'ocra1_vz', 'ocra1_vz2']:
                # marcompile will figure out whether the key matches the selected grad board
                keyb, channel = self.gradb.key_convert(key)
                keybin = keyb, # tuple
                valbin = self.gradb.float2bin(vals, channel),

                # hack to de-synchronise GPA-FHDO outputs, to give the
                # user the illusion of being able to output on several
                # channels in parallel
                if self._gpa_fhdo_offset_time:
                    tbin = times_us(times + channel*self._gpa_fhdo_offset_time + self._initial_wait),

            elif key in ['rx0_rate', 'rx1_rate']:
                keybin = key,
                valbin = vals.astype(np.uint16),
            elif key in ['rx0_rate_valid', 'rx1_rate_valid', 'rx0_rst_n', 'rx1_rst_n', 'rx0_en', 'rx1_en', 'tx_gate', 'rx_gate', 'trig_out']:
                keybin = key,
                # binary-valued data
                valbin = vals.astype(np.int32),
                for vb in valbin:
                    assert np.all( (0 <= vb) & (vb <= 1) ), "Binary columns must be [0,1] or [False, True] valued"
            elif key in ['leds']:
                keybin = key,
                valbin = vals.astype(np.int32),
            elif key in ['lo0_freq', 'lo1_freq', 'lo2_freq']:
                if np.issubdtype(vals.dtype, np.floating) or np.issubdtype(vals.dtype, np.integer) :
                    valbin = lo_real(vals),
                    keybin = key,
                else: 
                    assert False, f"The lo_freq is not a np.floating type."# Phase offset setting is not available so far.
            elif key in ['lo0_freq_offset', 'lo1_freq_offset', 'lo2_freq_offset']:
                lo_index = int(key[2]) 
                if np.issubdtype(vals.dtype, np.floating):
                    valbin = lo_real(vals+self._lo_freqs[lo_index]),
                    keybin = key[0:8],
                else: 
                    assert False, f"The lo_freq is not a np.floating type."# Phase offset setting is not available so far.
            elif key in ['lo0_rst', 'lo1_rst', 'lo2_rst']:
                valbin = vals.astype(np.int32),
                keybin = key,
            else:
                warnings.warn("Unknown marga experiment dictionary key: " + key)
                continue

            for t, k, v in zip(tbin, keybin, valbin):
                intdict[k] = (t, v)

        return intdict
    
if __name__ == "__main__":
    if True:
        expt = ExperimentMultiFreq(lo_freq=10.36, rx_t=3.125, allow_user_init_cfg=True)
        expt.add_flodict({"rx0_en": (np.array([200, 400, 1200, 1400]), np.array([1, 0, 1, 0]))})
        expt.add_flodict( {"tx0": (np.array([ 50, 130, 1000050,1130000]), np.array([ 0.5, 0,0.5, 0]))})
        # you can use either lo0_freq (to set frequency directly) or l00_freq_offset (to set frequency offset relative to the init lo_freq frequency)
        # expt.add_flodict( {"lo0_freq": (np.array([1000]), np.array([10.35]))})
        expt.add_flodict( {"lo0_freq_offset": (np.array([1000000]), np.array([-0.01]))}) # offset is -0.01MHz
        expt.add_flodict( {"lo0_rst": (np.array([10000,1000000+1/fpga_clk_freq_MHz]), np.array([1, 0]))})
        
        # expt.plot_sequence()
        # plt.show()
        rxd, msgs = expt.run()
