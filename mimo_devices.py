#!/usr/bin/env python3
#
# Utility class to operate MIMO systems

import queue
import socket, time, warnings
import threading
from contextlib import closing
import numpy as np
import matplotlib.pyplot as plt

import local_config as lc
import grad_board as gb
import server_comms as sc
import marcompile as mc

from device import Device
from acquisition import AcquisitionAborted, AcquisitionSession, AcquisitionState


def mimo_dev_run(devt):
    """Allows the parallelisation of Device run() calls, which is essential
    since the slaves will block otherwise."""
    dev, delay = devt
    if dev._seq is None:
        return {}, {'warning': 'No sequence execution elements, sequence not executed'}
    time.sleep(delay)
    rxd, msgs = dev.run()
    return rxd, msgs


class MimoDevices:
    """Manages multiple Devices, collates/distributes settings between them and
    handles messages and warnings."""

    # Subclasses may preserve site-specific Device behaviour while reusing the
    # synchronisation and lifecycle implementation below.
    device_class = Device

    def __init__(self, ips, ports, trig_output_time=5e3, slave_trig_latency=6.079,
                 trig_timeout=-1, master_run_delay=0, extra_args=None,
                 session=None, armed_timeout=30.0, **kwargs):
        """ips: list of device IPs, master first

        ports: list of device ports, master first

        trig_output_time: usec, when to trigger the slaves after beginning the
        sequence on the master

        trig_latency: usec, how long the slaves take from being triggered by the
        master to beginning their sequences (plus any additional I/O or cable
        latencies)

        trig_timeout: retained for source compatibility. Multi-board slaves now
        always use ``TRIG_FOREVER``; only an explicit trigger or emergency stop
        may release them.

        master_run_delay: deprecated compatibility argument. The master is now
        started from the server-confirmed armed barrier instead of a host-side
        sleep.

        extra_args: list of dictionaries of extra arguments to each Device
        object, master first

        All remaining arguments supported by the Device class are also
        supported, and will be passed down to each Device.

        """

        devN = len(ips)
        assert devN > 0, "At least one device is required"
        assert len(ips) == len(ports), f" Supplied {len(ips)} IPs but {len(ports)} ports"

        if extra_args is None:
            device_args = [kwargs] * devN
        else:
            assert len(ips) == len(extra_args), f" Supplied {len(ips)} IPs but {len(ports)} extra arg dicts"
            device_args = list(ea | kwargs for ea in extra_args)

        if master_run_delay:
            warnings.warn(
                "master_run_delay is ignored; MIMO now waits for confirmed slave armed events",
                DeprecationWarning,
                stacklevel=2,
            )
        if devN > 1 and trig_timeout >= 0:
            warnings.warn(
                "Finite slave trigger timeouts are disabled; using TRIG_FOREVER for safe MIMO arming",
                RuntimeWarning,
                stacklevel=2,
            )

        self._devs = []
        self._pool_args = []
        self._session = session if session is not None else AcquisitionSession()
        self._armed_timeout = float(armed_timeout)
        if self._armed_timeout <= 0:
            raise ValueError("armed_timeout must be positive")

        for k, (ip, port, devargs) in enumerate(zip(ips, ports, device_args)):
            # Device's current development default permits no-hardware timing
            # runs.  A MimoDevices acquisition is a hardware path unless the
            # caller explicitly supplied a previous/fake socket.
            devargs = dict(devargs)
            devargs.setdefault("prev_socket", None)

            if devN == 1:
                kwargs = devargs | {
                    'mimo_master': False,
                    'trig_timeout': 0,
                    'trig_output_time': 0,
                }
            elif k == 0:
                # TODO cannot yet handle the case where the MIMO system is externally triggered
                kwargs = devargs | {
                    'mimo_master': True,
                    'trig_timeout': 0,
                    'trig_output_time': trig_output_time }
            else:
                kwargs = devargs | {
                    'mimo_master': False,
                    'trig_timeout': -1,
                }

            dev = self.device_class(
                ip_address=ip, port=port, **kwargs)
            # Board 0 owns the RX gain bus independently of MIMO sync. Keep
            # this out of constructor kwargs for legacy Device adapters.
            dev._rx_gain_control = (k == 0)

            self._devs.append(dev)
            self._pool_args.append((dev, 0))

    def get_device(self, k):
        return self._devs[k]

    # shorthand
    def dev(self, k):
        return self.get_device(k)

    def dev_list(self):
        return self._devs

    @property
    def session(self):
        return self._session

    def _prepare_session(self, session):
        state = session.state
        if state == AcquisitionState.IDLE:
            session.begin()
        elif state in {
            AcquisitionState.FINISHED,
            AcquisitionState.ABORTED,
            AcquisitionState.FAILED,
        }:
            raise RuntimeError(
                f"Cannot run hardware from terminal acquisition state {state.value}; call begin() first"
            )
        elif state not in {AcquisitionState.PREPARING, AcquisitionState.RUNNING}:
            raise RuntimeError(
                f"Cannot start another hardware run while state is {state.value}"
            )

    @staticmethod
    def _advance_to_running(session, on_started=None):
        session.transition(AcquisitionState.RUNNING)
        if on_started is not None:
            on_started()

    @staticmethod
    def _inactive_result():
        return ({}, {"warning": "No sequence execution elements, sequence not executed"})

    def _run_single(self, session, on_started=None):
        session.transition(AcquisitionState.STARTING_MASTER)
        dev = self._devs[0]
        session.register_board(
            0, dev._ip_address, dev._port, dev._grad_board
        )
        session.checkpoint()
        result = dev.run(
            rearm=True,
            on_started=lambda: self._advance_to_running(session, on_started),
            send_context=session.send_guard,
            acquisition_id=session.acquisition_id,
        )
        session.checkpoint()
        results = [self._inactive_result() for _ in self._devs]
        results[0] = result
        return results

    def _run_multiple(self, session, active_board_ids, on_started=None):
        with closing(self._run_multiple_events(session, active_board_ids, on_started)) as events:
            for kind, payload in events:
                if kind == "done":
                    return payload

    def _run_multiple_events(self, session, active_board_ids, on_started=None,
                             rx_channel=None, chunk_points=None):
        """Share the armed barrier between ordinary runs and one RX stream."""
        session.transition(AcquisitionState.ARMING_SLAVES)
        for board_id in active_board_ids:
            dev = self._devs[board_id]
            session.register_board(
                board_id, dev._ip_address, dev._port, dev._grad_board
            )
        session.checkpoint()

        events = queue.Queue(maxsize=8 if rx_channel is not None else 0)
        threads = []
        closing_workers = threading.Event()

        def post_event(event):
            while not closing_workers.is_set():
                try:
                    events.put(event, timeout=0.05)
                    return
                except queue.Full:
                    pass

        def board_worker(board_id, dev):
            try:
                if board_id == 0:
                    callbacks = {
                        "on_started": lambda: self._advance_to_running(session, on_started)
                    }
                else:
                    callbacks = {
                        "report_armed": True,
                        "on_armed": lambda: post_event(("armed", board_id, None)),
                    }
                kwargs = dict(
                    send_context=session.send_guard,
                    acquisition_id=session.acquisition_id,
                    **callbacks,
                )
                if rx_channel is not None and board_id == rx_channel // 2:
                    physical_key = f"rx{rx_channel % 2}"
                    for rxd, msgs, meta in dev.run_stream(
                            chunk_points, [rx_channel % 2], **kwargs):
                        session.checkpoint()
                        if physical_key not in rxd and not meta.get("final"):
                            raise RuntimeError(f"RX chunk is missing {physical_key} on board {board_id}")
                        data = {f"rx{rx_channel}": rxd[physical_key]} if physical_key in rxd else {}
                        post_event(("chunk", board_id, (data, msgs, meta)))
                        if closing_workers.is_set():
                            return
                    result = ({}, {})
                else:
                    result = dev.run(rearm=True, **kwargs)
                post_event(("done", board_id, result))
            except BaseException as exc:
                post_event(("error", board_id, exc))

        def start_board(board_id):
            dev = self._devs[board_id]
            session.checkpoint()
            thread = threading.Thread(
                target=board_worker,
                args=(board_id, dev),
                name=f"marga-board-{board_id}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        complete = False
        try:
            slave_board_ids = active_board_ids[1:]
            for board_id in slave_board_ids:
                start_board(board_id)

            session.transition(AcquisitionState.WAITING_ALL_ARMED)
            waiting = set(slave_board_ids)
            deadline = time.monotonic() + self._armed_timeout
            while waiting:
                session.checkpoint()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for MIMO slaves to arm: {sorted(waiting)}"
                    )
                try:
                    kind, board_id, payload = events.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
                session.checkpoint()
                if kind == "armed":
                    waiting.discard(board_id)
                elif kind == "error":
                    raise payload
                else:
                    raise RuntimeError(
                        f"MIMO slave {board_id} completed before the master started"
                    )

            session.checkpoint()
            session.transition(AcquisitionState.STARTING_MASTER)
            results = [self._inactive_result() for _ in self._devs]
            start_board(0)
            completed = set()

            while len(completed) < len(active_board_ids):
                session.checkpoint()
                try:
                    kind, board_id, payload = events.get(timeout=0.05)
                except queue.Empty:
                    continue
                session.checkpoint()
                if kind == "error":
                    raise payload
                if kind == "chunk":
                    yield kind, payload
                if kind == "done":
                    results[board_id] = payload
                    completed.add(board_id)

            complete = True
            yield "done", results
        finally:
            closing_workers.set()
            if rx_channel is not None and not complete:
                # Also covers a caller closing the stream before its final reply.
                if session.cancelled:
                    session.ensure_hardware_stopped()
                else:
                    session.fail(stop_hardware=True)
                # Release blocked data sockets after the independent Stop requests.
                for board_id in active_board_ids:
                    sock = getattr(self._devs[board_id], "_s", None)
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except (AttributeError, OSError):
                            pass
            for thread in threads:
                thread.join(timeout=1.0 if rx_channel is not None else 0)

    def run(self, session=None, on_started=None):
        """Run all boards with a confirmed armed barrier for the MIMO case.

        Lifecycle finalisation is intentionally left to the owning sequence:
        one sequence may perform multiple batches/scans with the same session.
        ``on_started`` runs after the master's complete request has been sent
        (and, for MIMO, after all slaves confirmed armed). It is not an FPGA
        execution acknowledgement.
        """
        session = self._session if session is None else session
        self._session = session
        self._prepare_session(session)
        try:
            active_board_ids = [
                board_id
                for board_id, dev in enumerate(self._devs)
                if dev._seq is not None
            ]
            if not active_board_ids:
                return [self._inactive_result() for _ in self._devs]
            if active_board_ids[0] != 0:
                raise ValueError(
                    "MIMO board 0 must have sequence data whenever a slave board is active"
                )
            if len(active_board_ids) == 1:
                # A machine may have extra boards configured while a scan uses
                # only logical RX0/1.  Remove the master timing offset before
                # compilation and retain true direct-run semantics.
                self._devs[0].disable_mimo_sync()
            elif not self._devs[0]._mimo_master:
                raise RuntimeError(
                    "Slave boards became active after direct-run timing was selected; "
                    "create a new MimoDevices instance for this board set"
                )

            # Complete all compilation before registering any hardware board;
            # Stop during this phase is therefore a local cooperative cancel.
            for board_id in active_board_ids:
                session.checkpoint()
                self._devs[board_id].prepare()
            session.checkpoint()

            if len(active_board_ids) == 1:
                return self._run_single(session, on_started)
            return self._run_multiple(session, active_board_ids, on_started)
        except AcquisitionAborted:
            session.ensure_hardware_stopped()
            raise
        except Exception:
            # A slave may already be blocked in TRIG_FOREVER.  Always attempt
            # independent emergency stops before exposing the failure.
            session.fail(stop_hardware=True)
            raise

    def emergency_stop(self, timeout=2.0):
        return self._session.stop(timeout)

    def run_stream(self, chunk_points=None, rx_channels=None, session=None, on_started=None):
        """Stream RX data back in chunks.

        Yields ``(rxd_iq, msgs, meta)`` per chunk (same interface as
        ``Device.run_stream``). Multiple boards may execute, but exactly one
        logical RX channel must be selected in that case. Single-board channel
        selection is unchanged. RX names in yielded chunks are logical names.
        ``on_started`` has the same send-completion boundary as :meth:`run`.
        """
        rx_channel = None
        if len(self._devs) > 1:
            if rx_channels is None or len(rx_channels) != 1:
                raise NotImplementedError("Multi-board chunked streaming requires exactly one logical RX channel")
            rx_channel = int(rx_channels[0])
            if rx_channel != rx_channels[0] or not 0 <= rx_channel < 2 * len(self._devs):
                raise ValueError("RX channel must identify a configured board and physical channel 0/1")

        session = self._session if session is None else session
        self._session = session
        self._prepare_session(session)
        dev = self._devs[0]
        try:
            if rx_channel is not None:
                active_board_ids = [i for i, board in enumerate(self._devs) if board._seq is not None]
                if 0 not in active_board_ids or rx_channel // 2 not in active_board_ids:
                    raise ValueError("The master and selected RX board must both have sequence data")
                if len(active_board_ids) > 1:
                    if not dev._mimo_master:
                        raise RuntimeError("Create a new MimoDevices instance before activating slave boards")
                    for board_id in active_board_ids:
                        session.checkpoint()
                        self._devs[board_id].prepare()
                    final_chunk = None
                    with closing(self._run_multiple_events(
                            session, active_board_ids, on_started,
                            rx_channel, chunk_points)) as events:
                        for kind, payload in events:
                            session.checkpoint()
                            if kind == "chunk":
                                if payload[2].get("final"):
                                    final_chunk = payload
                                else:
                                    yield payload
                    if final_chunk is None:
                        raise ConnectionError("RX stream ended without its final reply")
                    session.checkpoint()
                    yield final_chunk  # Success only after every participating board finished.
                    return
                dev.disable_mimo_sync()
            session.checkpoint()
            dev.prepare()
            session.transition(AcquisitionState.STARTING_MASTER)
            session.register_board(
                0, dev._ip_address, dev._port, dev._grad_board
            )
            session.checkpoint()
            yield from dev.run_stream(
                chunk_points,
                rx_channels,
                on_started=lambda: self._advance_to_running(session, on_started),
                send_context=session.send_guard,
                acquisition_id=session.acquisition_id,
            )
            session.checkpoint()
        except AcquisitionAborted:
            session.ensure_hardware_stopped()
            raise
        except Exception:
            session.fail(stop_hardware=True)
            raise


def test_mimo_devices(single=True, reps=1, **kwargs):
    # importing here to avoid a circular import - these are just throwaway plotting functions
    from test_mimo_lowlevel import plot_single, plot_repeated

    ips = [lc.ip_address, lc.ip_address_slave]
    ports = [lc.port, lc.port_slave]

    # Manually handle sockets if repeated tests are being run
    socks = []
    for ip, port in zip(ips, ports):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        socks.append(sock)

    # How to do per-Device extra arguments

    # either
    extra_args = list({'prev_socket': sock } for sock in socks)

    # or
    extra_args = [{'prev_socket': socks[0]}, {'prev_socket': socks[1]}]

    # How to do global arguments (shared for every device)
    rx_t = 1

    mdev = MimoDevices(ips=ips, ports=ports, extra_args=extra_args, rx_t=rx_t, **kwargs)
    devs = mdev.dev_list()

    for dev in devs:
        dev.add_flodict({
            'tx0': (np.array([10, 30, 80, 90]), np.array([0.3+0.5j, 0, 0.8+0.3j, 0])),
            'tx1': (np.array([20, 40, 75, 85]), np.array([0.3+0.5j, 0, 0.4+0.2j, 0])),
            'rx0_en': (np.array([0, 100]), np.array([1, 0])),
            'rx1_en': (np.array([0, 100]), np.array([1, 0])),
            })

    if single:
        res = mdev.run()
        plot_single(res, rx_t)
        assert reps == 1, "reps must be 1 for single test"
    else:
        resl = []
        for k in range(reps):
            print(f"Round {k+1}")
            res = mdev.run()
            resl.append(res)

        plot_repeated(resl, rx_t, cross_only=True)

    plt.show()
    # Only necessary if the socket isn't being closed anyway
    (dev.__del__() for dev in devs)


if __name__ == "__main__":
    ## Single test
    # test_mimo_devices(single=True)

    ## 100x repetitions, default trig output time, default trig timeout
    test_mimo_devices(single=False, reps=100)

    ## 5x repetitions, 5-ms trigger lead, infinite trigger wait
    # test_mimo_devices(single=False, reps=5, trig_output_time=5e3, trig_timeout=-1)
