#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy, time
import multiprocessing as mp

from device import Device
import server_comms as sc
import test_base as tb


def mimo_dev_run(devt):
    """Allows the parallelisation of Device run() calls, which is essential
    since the slaves will block otherwise."""
    dev, delay = devt
    time.sleep(delay)
    rxd, msgs = dev.run()
    return rxd, msgs


def test_mimo_lowlevel(
    sim=False, sock_m=None, sock_s=None, plot_preview=False, plot_data=True
):
    """Manual 2-board master-slave synchronisation test. Main steps are:
    - start master manually
    - start slave (immediate trigger wait)
    - get master data manually

    Both boards have detection windows on RX0 and RX1, and single RF pulses on
    TX0 and TX1. If we label boards 0/1 as B0 and B1, you should connect them as follows:

    B0_TX0 to B0_RX0
    B0_TX1 to B1_RX1
    B1_TX0 to B1_RX0
    B1_TX1 to B0_RX1

    The returned data should show the same signal on all four channels, down to
    single-cycle timing accuracy. Amplitudes/phases could of course vary,
    depending on the cables you use and fine differences between the channels,
    but the jitter should be below one cycle.
    """

    if sim:
        master_ip = "localhost"
        slave_ip = "localhost"
        master_port = 11111
        slave_port = 11112
    else:
        master_ip = "192.168.1.158"
        slave_ip = "192.168.1.160"
        master_port = 11111
        slave_port = 11111

    # when to pulse the output trigger after the start of the master sequence
    trig_time = 1000

    if sim:
        trig_wait_time = 1000
    else:
        # cycles for slave to wait for trigger arrival before giving up
        trig_wait_time = 16e6

    start_offset = 20.0
    rx_gate_len = 200.0
    rf_pulse_frac = 0.5  # fraction of rx_gate_len
    tx0_amp = 0.5
    tx1_amp = 0.4

    dev_kwargs = {
        "lo_freq": 2.5,
        "rx_t": 3.125,
        "print_infos": True,
        "assert_errors": True,
        "halt_and_reset": False,
    }

    dev_m = Device(
        ip_address=master_ip, port=master_port, prev_socket=sock_m, **dev_kwargs
    )
    dev_s = Device(
        ip_address=master_ip,
        port=slave_port,
        prev_socket=sock_s,
        trig_wait_time=trig_wait_time,
        **dev_kwargs
    )

    slave_tx_t = start_offset + np.array(
        [
            0.5 * (1 - rf_pulse_frac) * rx_gate_len,
            0.5 * (1 + rf_pulse_frac) * rx_gate_len,
        ]
    )

    slave_rx_t = start_offset + np.array([0, rx_gate_len])

    slave_fd = {
        "tx0": (slave_tx_t, np.array([tx0_amp, 0])),
        "tx1": (slave_tx_t, np.array([tx1_amp, 0])),
        "rx0_en": (slave_rx_t, np.array([1, 0])),
        "rx1_en": (slave_rx_t, np.array([1, 0])),
    }

    master_fd = dict(slave_fd)
    master_fd["trig_out"] = (np.array([trig_time]), np.array([1]))
    for key in ["tx0", "tx1", "rx0_en", "rx1_en"]:
        # shift times by trigger delay
        master_fd[key] = (master_fd[key][0] + trig_time, master_fd[key][1])

    dev_m.add_flodict(master_fd)
    dev_s.add_flodict(slave_fd)

    if plot_preview:
        dev_m.plot_sequence()
        dev_s.plot_sequence()
        plt.show()

    mpl = [(dev_s, 0), (dev_m, 2)]

    with mp.Pool(2) as p:
        res = p.map(mimo_dev_run, mpl)

    for dev, _ in mpl:
        dev.close_server(only_if_sim=True)

    if plot_data:
        for rxd, msgs in res:
            plt.plot(np.abs(rxd["rx0"]) + np.random.random_sample())
            plt.plot(np.abs(rxd["rx1"]) + np.random.random_sample())
            print(msgs)

        plt.show()


if __name__ == "__main__":
    sim = True

    if sim:
        tb.base_setup_class()
        pm, sm = tb.base_setup(  # master
            fst_dump=False, csv_path=os.path.join("/tmp", "marga_sim_m.csv"), port=11111
        )
        ps, ss = tb.base_setup(  # slave
            fst_dump=False, csv_path=os.path.join("/tmp", "marga_sim_s.csv"), port=11112
        )

        test_mimo_lowlevel(sim=True, sock_m=sm, sock_s=ss)

        # halt simulation
        sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sm)
        sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), ss)
        sm.close()
        ss.close()
        pm.wait(1)  # wait a short time for simulator to close
        ps.wait(1)
    else:
        test_mimo_lowlevel()
