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
        
        def deg2ddsPhase(farr):
            dds_phase = np.round(1 / 360 * 2**31 * np.array(farr)).astype(np.uint32)
            return dds_phase
        
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
                if np.issubdtype(vals.dtype, np.floating) or np.issubdtype(vals.dtype, np.integer):
                    valbin = lo_real(vals+self._lo_freqs[lo_index]),
                    keybin = key[0:8],
                else: 
                    assert False, f"The lo_freq is not a np.floating type."# Phase offset setting is not available so far.
            elif key in ['lo0_rst', 'lo1_rst', 'lo2_rst']:
                valbin = vals.astype(np.int32),
                keybin = key,
            elif key in ['lo0_phase_rst', 'lo1_phase_rst', 'lo2_phase_rst']:
                valbin = vals.astype(np.int32),
                keybin = key,
            elif key in ['lo0_phase', 'lo1_phase', 'lo2_phase']:
                valbin = deg2ddsPhase(vals),
                keybin = key[0:4]+'freq',
            else:
                warnings.warn("Unknown marga experiment dictionary key: " + key)
                continue

            for t, k, v in zip(tbin, keybin, valbin):
                if k in intdict:
                    # Append new data to the existing NumPy array
                    intdict[k] = (np.concatenate((intdict[k][0], t)), 
                                  np.concatenate((intdict[k][1], v)))
                else:
                    # Initialize with a tuple of arrays if key does not exist
                    intdict[k] = (t, v)

        return intdict
    
# Main script for configuring and running a multi-frequency experiment
if __name__ == "__main__":
    # Initialize the experiment with basic parameters
    expt = ExperimentMultiFreq(
        lo_freq=10.36,  # Local oscillator base frequency in MHz
        rx_t=3.125,     # ADC dwell time in us
        allow_user_init_cfg=True  # Allow user to initialize custom configurations including lo*_freq
    )
    # Demo
    if False:
        # Add receiver (rx) configurations
        # rx0_en defines time points and enabling/disabling states
        expt.add_flodict({
            "rx0_en": (
                np.array([200, 400, 1200, 1400]),  # Time points in us
                np.array([1, 0, 1, 0])            # Enable (1) or disable (0)
            )
        })

        # Add transmitter (tx) configurations
        # tx0 defines time points and amplitude values
        expt.add_flodict({
            "tx0": (
                np.array([50, 130, 1000050, 1130000]),  # Time points in us
                np.array([0.5, 0, 0.5, 0])             # Amplitude values
            )
        })

        ###########################################################
        ### Local oscillator (LO) frequency and offset settings ###
        # Method 1: Direct frequency setting (uncomment if needed)
        # expt.add_flodict({
        #     "lo0_freq": (
        #         np.array([1000]),  # Time point in us
        #         np.array([10.35])  # Frequency in MHz
        #     )
        # })

        # Method 2: Frequency offset setting (relative to lo_freq)
        expt.add_flodict({
            "lo0_freq_offset": (
                np.array([1000000]),     # Time point in us
                np.array([-0.01])       # Offset in MHz
            )
        })

        # Reset LO phase using lo0_rst
        expt.add_flodict({
            "lo0_rst": (
                np.array([1000000, 1000000 + 1 / fpga_clk_freq_MHz]),  # Time points in us
                np.array([1, 0])                                       # Reset state
            )
        })
        ###########################################################
        ### Local oscillator phase settings ###
        # Ensure the FPGA is loaded with the appropriate phase-written version of the binary file.
        # Refer to the "linkaishengmri/marcos_extras" repository for binaries.

        # Phase reset configuration (lo0_phase_rst)
        expt.add_flodict({
            "lo0_phase_rst": (
                np.array([2000000, 2000000 + 1 / fpga_clk_freq_MHz]),  # Time points in us
                np.array([1, 0])                                       # Reset state
            )
        })

        # Set phase directly using lo0_phase
        # After setting phase_rst to 0, reconfigure lo0_freq_offset !!!
        expt.add_flodict({
            "lo0_phase": (
                np.array([2000000]),  # Time point in us
                np.array([180])       # Phase in degrees
            ),
            "lo0_freq_offset": (
                np.array([2000000 + 1 / fpga_clk_freq_MHz]),  # Time point in us
                np.array([-0.01])                             # Offset in MHz
            )
        })
        ###########################################################

    if True:
        # Test phase
        # Time(+10000us)     Events:       
        # 0 us               Tx begin with some phase.
        # 1 us               Set freq with 0 phase. (lo_rst:1)
        # 2 us               Set phase to 0 deg. (lo_phase:1, reconfig freq_offset)
        # 3 us               
        # 4 us
        # 5 us               End of Tx pulse
        expt.add_flodict({"tx0": (np.array([10000, 10005]), np.array([0.2, 0])),
                         "lo0_freq_offset":(np.array([10001, 10002+1/fpga_clk_freq_MHz]), np.array([-0.01, -0.01])),
                         "lo0_rst":(np.array([10001, 10001+1/fpga_clk_freq_MHz]), np.array([1, 0])),
                         "lo0_phase": (np.array([10002]), np.array([0])),
                         "lo0_phase_rst": (np.array([10002, 10002+1/fpga_clk_freq_MHz]), np.array([1, 0])),
                         })

    # Optional: Visualize the sequence (uncomment if needed)
    # expt.plot_sequence()
    # plt.show()

    # Run the experiment and collect data
    rxd, msgs = expt.run()

