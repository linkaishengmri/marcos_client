#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy, time, warnings
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
        master_ip = "192.168.1.160"
        slave_ip = "192.168.1.158"
        master_port = 11111
        slave_port = 11111

    # when to pulse the output trigger after the start of the master sequence
    trig_time = 200e3

    # how long the slaves take to get triggered by the master, usec
    trig_latency = 6.08

    if sim:
        trig_wait_time = 1000
    else:
        # cycles for slave to wait for trigger arrival before giving up
        # trig_wait_time = int(16e6)
        # trig_wait_time = 0
        trig_wait_time = -1

    start_offset = 0

    rx_gates = 20
    rx_gate_interval = 1000e3  # time to wait between gates

    rx_gate_len = 2.0  # 1us
    rf_pulse_frac = 0.5  # fraction of rx_gate_len
    rf_pulse_offset = -0.15  # fraction of rx_gate_len to move
    tx0_amp = 0.5 + 0.3j
    tx1_amp = tx0_amp

    lo_freq = 20
    rx_t = 0.0326

    dev_kwargs = {
        # "lo_freq": 2.5,
        "lo_freq": lo_freq,
        "rx_t": rx_t,
        "print_infos": True,
        "assert_errors": True,
        "halt_and_reset": False,
        "fix_cic_scale": True,
        "set_cic_shift": False,  # needs to be true for open-source cores
        "flush_old_rx": False,
    }

    dev_m = Device(
        ip_address=master_ip, port=master_port, prev_socket=sock_m, **dev_kwargs
    )
    dev_s = Device(
        ip_address=slave_ip,
        port=slave_port,
        prev_socket=sock_s,
        trig_wait_time=trig_wait_time,
        **dev_kwargs,
    )

    slave_tx_t = np.zeros(2 * rx_gates)
    slave_rx_t = np.zeros_like(slave_tx_t)
    slave_tx0_amp = np.zeros_like(slave_tx_t, dtype=complex)
    slave_tx1_amp = np.zeros_like(slave_tx_t, dtype=complex)
    slave_rx0_en = np.zeros_like(slave_tx_t, dtype=int)
    slave_rx1_en = np.zeros_like(slave_tx_t, dtype=int)

    slave_tx_t_start = (0.5 * (1 - rf_pulse_frac) + rf_pulse_offset) * rx_gate_len
    slave_tx_t_end = (0.5 * (1 + rf_pulse_frac) + rf_pulse_offset) * rx_gate_len
    for gate in range(rx_gates):
        gate_start = start_offset + gate * (rx_gate_len + rx_gate_interval)
        slave_tx_t[2 * gate] = gate_start + slave_tx_t_start
        slave_tx_t[2 * gate + 1] = gate_start + slave_tx_t_end
        slave_tx0_amp[2 * gate] = tx0_amp
        slave_tx1_amp[2 * gate] = tx1_amp

        slave_rx_t[2 * gate] = gate_start
        slave_rx_t[2 * gate + 1] = gate_start + rx_gate_len
        slave_rx0_en[2 * gate] = 1
        slave_rx1_en[2 * gate] = 1

    slave_fd = {
        "tx0": (slave_tx_t, slave_tx0_amp),
        "tx1": (slave_tx_t, slave_tx1_amp),
        "tx_gate": (
            np.array([trig_time * 2, trig_time * 3]),
            np.array([1, 0]),
        ),  # just to have something at the end of the sequence
        "rx0_en": (slave_rx_t, slave_rx0_en),
        "rx1_en": (slave_rx_t, slave_rx1_en),
    }

    master_fd = dict(slave_fd)

    # 1us trig pulse
    master_fd["trig_out"] = (np.array([trig_time, trig_time + 1]), np.array([1, 0]))

    for key in ["tx0", "tx1", "rx0_en", "rx1_en"]:
        # shift times by trigger time and trigger latency
        master_fd[key] = (
            master_fd[key][0] + trig_time + trig_latency,
            master_fd[key][1],
        )

    dev_m.add_flodict(master_fd)
    dev_s.add_flodict(slave_fd)

    if plot_preview:
        plt.subplot(211)
        dev_m.plot_sequence()
        plt.subplot(212)
        dev_s.plot_sequence()
        plt.show()

    mpl = [(dev_m, 0), (dev_s, 0)]
    # mpl = [(dev_m, 0)]

    with mp.Pool(2) as p:
        res = p.map(mimo_dev_run, mpl)

    for dev, _ in mpl:
        dev.close_server(only_if_sim=True)

    if plot_data:
        if False:
            for rxd, msgs in res:
                print(msgs)

                # Phase-sensitive, overlapping
                plt.subplot(211)
                try:
                    plt.plot(np.real(rxd["rx0"]))
                    plt.plot(np.imag(rxd["rx0"]))
                except KeyError:
                    pass
                plt.subplot(212)
                try:
                    plt.plot(np.real(rxd["rx1"]))
                    plt.plot(np.imag(rxd["rx1"]))
                except KeyError:
                    pass
                # plt.plot(np.abs(rxd["rx0"]) + np.random.random_sample())
                # plt.plot(np.abs(rxd["rx1"]) + np.random.random_sample())

        else:
            # Assume master is first in the list, slaves are 2nd and thereafter
            rxdm, msgsm = res[0]
            rxds, msgss = res[1]
            rx_len = 0
            for k in [rxdm, rxds]:
                for s in ["rx0", "rx1"]:
                    if rx_len == 0:
                        rx_len = len(k[s])
                    elif rx_len != len(k[s]):
                        warnings.warn("RX lengths not all equal -- check timings!")

            plt.figure(figsize=(14, 13))
            # Phases
            plt.subplot(211)
            plt.title(
                f"{rx_gates} RX gates + TX pulses, {rx_gate_interval/1e3:.2f}ms gate interval, {rx_gate_len:.2f}us gate length, {lo_freq:.2f}MHz LO freq, {1/rx_t:.2f}MHz sample rate"
            )
            plt.plot(np.real(rxdm["rx0"]), label="rx0_real")
            plt.plot(np.imag(rxdm["rx0"]), label="rx0_imag")
            plt.ylabel("Master RX0")
            plt.legend()
            plt.grid(True)
            plt.subplot(212)
            plt.ylabel("Slave RX1")
            plt.plot(np.real(rxds["rx1"]), label="rx1_real")
            plt.plot(np.imag(rxds["rx1"]), label="rx1_imag")
            plt.legend()
            plt.grid(True)

            plt.figure(figsize=(14, 13))
            # Absolute amplitudes
            plt.subplot(311)
            plt.title(
                f"{rx_gates} RX gates + TX pulses, {rx_gate_interval/1e3:.2f}ms gate interval, {rx_gate_len:.2f}us gate length, {lo_freq:.2f}MHz LO freq, {1/rx_t:.2f}MHz sample rate"
            )

            plt.plot(np.abs(rxdm["rx0"]), label="rx0")
            plt.plot(np.abs(rxdm["rx1"]), label="rx1")
            plt.ylabel("Master")
            plt.legend()
            plt.grid(True)
            plt.subplot(312)
            plt.ylabel("Slave")
            plt.plot(np.abs(rxds["rx0"]), label="rx0")
            plt.plot(np.abs(rxds["rx1"]), label="rx1")
            plt.legend()
            plt.grid(True)
            plt.subplot(313)
            plt.ylabel("Master RX0 - slave RX1")

            rxdm0_abs = np.abs(rxdm["rx0"])
            rxds1_abs = np.abs(rxds["rx1"])
            rxdm0_norm = rxdm0_abs / np.mean(rxdm0_abs)
            rxds1_norm = rxds1_abs / np.mean(rxds1_abs)
            plt.plot(rxdm0_abs - rxds1_abs, label="rx0_m - rx1_s")
            plt.plot(rxdm0_norm - rxds1_norm, label="norm(rx0_m) - norm(rx1_s)")
            plt.legend()
            plt.grid(True)

        plt.show()


if __name__ == "__main__":
    sim = False

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
        test_mimo_lowlevel(plot_preview=False)
