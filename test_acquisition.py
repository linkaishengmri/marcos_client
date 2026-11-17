#!/usr/bin/env python3

import socket
import threading
import time
import unittest
from unittest import mock

import msgpack
import numpy as np

from acquisition import AcquisitionAborted, AcquisitionSession, AcquisitionState
from device import Device
from mimo_devices import MimoDevices
import server_comms as sc


class _FakeDevice:
    calls = []
    master_started = threading.Event()

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.master_started = threading.Event()

    def __init__(self, ip_address, port, **kwargs):
        self._ip_address = ip_address
        self._port = port
        self._grad_board = kwargs.get("grad_board", "ocra1")
        self._mimo_master = kwargs.get("mimo_master", False)
        self._trig_output_time = kwargs.get("trig_output_time", 0)
        self._slave_trig_latency = kwargs.get("slave_trig_latency", 0)
        self._trig_wait_time = kwargs.get("trig_timeout", 0)
        self._master_started = type(self).master_started
        self._seq = {"fake": ([], [])}
        self._seq_compiled = False

    def prepare(self):
        if getattr(self, "prepare_error", None) is not None:
            raise self.prepare_error
        self._seq_compiled = True

    def disable_mimo_sync(self):
        self._mimo_master = False
        self._trig_output_time = 0
        self._slave_trig_latency = 0

    def run(
        self,
        report_armed=False,
        on_armed=None,
        on_started=None,
        rearm=False,
        send_context=None,
        acquisition_id=None,
    ):
        if send_context is not None:
            with send_context():
                pass
        self.calls.append(("send", self._ip_address, report_armed, rearm))
        if report_armed:
            time.sleep(getattr(self, "armed_delay", 0))
            self.calls.append(("armed", self._ip_address))
            on_armed()
            if getattr(self, "finish_before_master", False):
                return ({"rx0": []}, {})
            if not self._master_started.wait(1):
                raise TimeoutError("test master did not start")
        else:
            if on_started is not None:
                on_started()
            self.calls.append(("master_started", self._ip_address))
            self._master_started.set()
        return ({"rx0": []}, {})


class _FakeMimoDevices(MimoDevices):
    device_class = _FakeDevice


class _ReplySocket:
    def __init__(self, replies):
        self.sent = []
        self._chunks = [b"".join(msgpack.packb(reply) for reply in replies)]

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _BlockingReplySocket(_ReplySocket):
    def __init__(self, replies):
        super().__init__(replies)
        self.send_entered = threading.Event()
        self.release_send = threading.Event()

    def sendall(self, data):
        self.send_entered.set()
        self.release_send.wait(timeout=1)
        super().sendall(data)


class AcquisitionSessionTest(unittest.TestCase):
    def test_stop_while_preparing_is_local_and_latched(self):
        session = AcquisitionSession()
        session.begin()
        acquisition_id = session.acquisition_id
        for _ in range(3):
            session.request_cancel()
            self.assertEqual(session.stop(), {})
            self.assertEqual(session.state, AcquisitionState.ABORTED)
            self.assertEqual(session.acquisition_id, acquisition_id)
            with self.assertRaises(AcquisitionAborted):
                session.checkpoint()

    def test_emergency_stop_targets_all_boards_concurrently(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.ARMING_SLAVES)
        session.register_board(0, "master", 11111, "ocra1")
        session.register_board(1, "slave", 11111, "gpa-fhdo")
        barrier = threading.Barrier(2)

        def stop_board(board, timeout, acquisition_id):
            barrier.wait(timeout=1)
            return board.grad_board

        with mock.patch.object(
            AcquisitionSession, "_stop_board", side_effect=stop_board
        ):
            results = session.stop()

        self.assertEqual(results, {0: "ocra1", 1: "gpa-fhdo"})
        self.assertEqual(session.state, AcquisitionState.ABORTED)

    def test_unconfirmed_emergency_stop_marks_session_failed(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.register_board(0, "master", 11111, "ocra1")
        with mock.patch.object(
            AcquisitionSession,
            "_stop_board",
            side_effect=RuntimeError("safe-state acknowledgement missing"),
        ):
            results = session.stop()

        self.assertIsInstance(results[0], RuntimeError)
        self.assertEqual(session.state, AcquisitionState.FAILED)
        self.assertEqual(
            session.snapshot()["stop_failures"],
            {0: "safe-state acknowledgement missing"},
        )

        with self.assertRaisesRegex(RuntimeError, "emergency-safe state"):
            session.begin()

        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ) as stop_board:
            self.assertEqual(session.stop(), {0: "stopped"})
            self.assertEqual(session.stop(), {0: "stopped"})
        stop_board.assert_called_once()
        self.assertEqual(session.state, AcquisitionState.FAILED)
        with self.assertRaises(AcquisitionAborted):
            session.checkpoint()
        session.begin()
        session.checkpoint()
        self.assertEqual(session.state, AcquisitionState.PREPARING)

    def test_terminal_stop_returns_to_idle(self):
        session = AcquisitionSession()
        session.begin()
        session.finish()
        session.stop()
        self.assertEqual(session.state, AcquisitionState.IDLE)

    def test_repeated_stop_after_confirmed_abort_stays_latched(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.register_board(0, "master", 11111, "ocra1")
        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ) as stop_board:
            self.assertEqual(session.stop(), {0: "stopped"})
            stopped = session.snapshot()
            for _ in range(3):
                session.request_cancel()
                self.assertEqual(session.stop(), {0: "stopped"})
                self.assertEqual(session.snapshot(), stopped)
                with self.assertRaises(AcquisitionAborted):
                    session.checkpoint()

        stop_board.assert_called_once()
        self.assertEqual(session.state, AcquisitionState.ABORTED)
        session.begin()
        session.checkpoint()
        self.assertEqual(session.state, AcquisitionState.PREPARING)
        self.assertNotEqual(session.acquisition_id, stopped["acquisition_id"])

    def test_stop_after_failure_preserves_failure_and_safe_confirmation(self):
        for already_stopped in (False, True):
            with self.subTest(already_stopped=already_stopped):
                session = AcquisitionSession()
                session.begin()
                session.transition(AcquisitionState.STARTING_MASTER)
                session.register_board(0, "master", 11111, "ocra1")
                acquisition_id = session.acquisition_id
                with mock.patch.object(
                    AcquisitionSession, "_stop_board", return_value="stopped"
                ) as stop_board:
                    session.fail(stop_hardware=already_stopped)
                    for _ in range(3):
                        self.assertEqual(session.stop(), {0: "stopped"})
                        self.assertEqual(session.state, AcquisitionState.FAILED)
                        self.assertEqual(session.acquisition_id, acquisition_id)
                        self.assertEqual(session.participating_board_ids, (0,))
                        with self.assertRaises(AcquisitionAborted):
                            session.checkpoint()
                stop_board.assert_called_once()
                session.begin()
                session.checkpoint()
                self.assertEqual(session.state, AcquisitionState.PREPARING)

    def test_cancel_latch_then_background_stop_still_dispatches(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.register_board(0, "master", 11111, "ocra1")
        session.request_cancel()
        with self.assertRaises(AcquisitionAborted):
            session.checkpoint()

        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ) as stop_board:
            self.assertEqual(session.stop(), {0: "stopped"})
        stop_board.assert_called_once()
        self.assertEqual(session.state, AcquisitionState.ABORTED)

    def test_begin_cannot_clear_pending_hardware_stop(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.register_board(0, "master", 11111, "ocra1")
        session.request_cancel()
        with self.assertRaises(RuntimeError):
            session.begin()

        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ):
            session.stop()
        old_id = session.acquisition_id
        session.begin()
        self.assertNotEqual(session.acquisition_id, old_id)
        self.assertEqual(session.state, AcquisitionState.PREPARING)

    def test_concurrent_stops_share_one_complete_dispatch(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.transition(AcquisitionState.RUNNING)
        session.register_board(0, "master", 11111, "ocra1")
        entered = threading.Event()
        release = threading.Event()
        results = []

        def stop_board(board, timeout, acquisition_id):
            entered.set()
            release.wait(timeout=1)
            return acquisition_id

        with mock.patch.object(
            AcquisitionSession, "_stop_board", side_effect=stop_board
        ) as stop_board_mock:
            first = threading.Thread(target=lambda: results.append(session.stop()))
            second = threading.Thread(target=lambda: results.append(session.stop()))
            first.start()
            self.assertTrue(entered.wait(1))
            second.start()
            time.sleep(0.01)
            self.assertEqual(session.participating_board_ids, (0,))
            release.set()
            first.join(1)
            second.join(1)

        self.assertEqual(stop_board_mock.call_count, 1)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], session.acquisition_id)
        self.assertEqual(session.state, AcquisitionState.ABORTED)


class RxGainTest(unittest.TestCase):
    gains = {0: 100, 1: 50, 2: 10, 3: 0}
    expected_writes = [(0, 0), (1, 12), (2, 40), (3, 63)]

    def setUp(self):
        # No constructor or run-path regression may open a real connection.
        guard = mock.patch("socket.socket", side_effect=AssertionError("Hardware I/O forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def make_devices(self, count=1, gains=None, clock=125.0):
        devices = MimoDevices(
            ips=["offline"] * count,
            ports=[11111] * count,
            prev_socket=object(),
            init_gpa=False,
            fpga_clk_freq_MHz=clock,
            grad_board="ocra1_5761",
            rx_gain=dict(self.gains if gains is None else gains),
            trig_output_time=10000,
            auto_leds=True,
            allow_user_init_cfg=True,
        )
        for dev in devices.dev_list():
            dev.add_flodict({
                "tx0": (np.array([1000., 2000.]), np.array([0.1 + 0.2j, 0j])),
                "rx0_en": (np.array([1000., 2000.]), np.array([1, 0])),
                "grad_vx": (np.array([1000., 2000.]), np.array([0.01, 0.])),
            })
        return devices

    @staticmethod
    def gain_writes(dev):
        """Read channel/code pairs latched at gain-write strobes."""
        registers = dict(zip(*dev._seq["rxgain_reg"]))
        address_times = [t for t, value in zip(*dev._seq["rxgain_sel"]) if value == 1]
        writes = []
        for time, value in zip(*dev._seq["rxgain_write"]):
            if value == 1:
                address_time = max(t for t in address_times if t < time)
                value_time = max(t for t in registers if t < time)
                writes.append((int(registers[address_time]), int(registers[value_time])))
        return writes

    def test_single_board_programs_all_gains_without_mimo_trigger(self):
        for clock in (122.88, 125.0):
            with self.subTest(clock=clock):
                dev = self.make_devices(clock=clock).dev(0)
                dev.prepare()
                self.assertEqual(self.gain_writes(dev), self.expected_writes)
                self.assertFalse(dev._mimo_master)
                self.assertEqual(dev._trig_wait_time, 0)
                self.assertNotIn("trig_out", dev._seq)

    def test_gain_edit_changes_machine_code_but_not_other_waveforms(self):
        low = self.make_devices(gains={0: 1, 1: 100}).dev(0)
        high = self.make_devices(gains={0: 100, 1: 100}).dev(0)
        low.prepare()
        high.prepare()
        self.assertFalse(np.array_equal(low._machine_code, high._machine_code))
        self.assertEqual(self.gain_writes(low), [(0, 63), (1, 0)])
        self.assertEqual(self.gain_writes(high), [(0, 0), (1, 0)])
        for key in low._seq:
            if not key.startswith("rxgain_"):
                for actual, expected in zip(low._seq[key], high._seq[key]):
                    np.testing.assert_array_equal(actual, expected, err_msg=key)

    def test_multiple_boards_only_program_gain_on_master(self):
        devices = self.make_devices(count=3)
        for dev in devices.dev_list():
            dev.prepare()
        self.assertEqual(self.gain_writes(devices.dev(0)), self.expected_writes)
        self.assertIn("trig_out", devices.dev(0)._seq)
        for dev in devices.dev_list()[1:]:
            self.assertEqual(self.gain_writes(dev), [])
            self.assertEqual(dev._trig_wait_time, -1)
            self.assertNotIn("trig_out", dev._seq)

    def test_inactive_slaves_preserve_gain_when_sync_is_disabled(self):
        devices = self.make_devices(count=2)
        master = devices.dev(0)
        devices.dev(1)._seq = None
        # Exercise the actual active-board selection and preparation in run().
        # Stop at the transport boundary, after compiling the real master.
        with mock.patch.object(devices, "_run_single", return_value=[]) as transport:
            devices.run()
        transport.assert_called_once()
        self.assertEqual(self.gain_writes(master), self.expected_writes)
        self.assertFalse(master._mimo_master)
        self.assertNotIn("trig_out", master._seq)
        direct = self.make_devices().dev(0)
        direct.prepare()
        np.testing.assert_array_equal(master._machine_code, direct._machine_code)

    def test_repeated_prepare_keeps_gain_writes_and_machine_code_stable(self):
        dev = self.make_devices().dev(0)
        dev.prepare()
        original = dev._machine_code.copy()
        dev.prepare()
        self.assertEqual(self.gain_writes(dev), self.expected_writes)
        np.testing.assert_array_equal(dev._machine_code, original)

    def test_direct_device_retains_legacy_gain_ownership(self):
        for master in (False, True):
            with self.subTest(master=master):
                dev = Device(prev_socket=object(), mimo_master=master,
                             rx_gain=dict(self.gains), auto_leds=False,
                             fpga_clk_freq_MHz=125.0, grad_board="ocra1_5761")
                dev.add_flodict({"rx0_en": (np.array([1000., 2000.]), np.array([1, 0]))})
                if master:
                    dev.disable_mimo_sync()
                dev.prepare()
                self.assertEqual(self.gain_writes(dev), self.expected_writes if master else [])
                self.assertNotIn("trig_out", dev._seq)


class MimoBarrierTest(unittest.TestCase):
    def setUp(self):
        _FakeDevice.reset()

    def test_all_slaves_arm_before_master_starts(self):
        states = []
        session = AcquisitionSession(states.append)
        devices = _FakeMimoDevices(
            ["master", "slave1", "slave2"],
            [11111, 11111, 11111],
            session=session,
        )
        result = devices.run()

        master_index = _FakeDevice.calls.index(("master_started", "master"))
        self.assertLess(
            _FakeDevice.calls.index(("armed", "slave1")), master_index
        )
        self.assertLess(
            _FakeDevice.calls.index(("armed", "slave2")), master_index
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(session.state, AcquisitionState.RUNNING)
        self.assertEqual(
            states,
            [
                AcquisitionState.PREPARING,
                AcquisitionState.ARMING_SLAVES,
                AcquisitionState.WAITING_ALL_ARMED,
                AcquisitionState.STARTING_MASTER,
                AcquisitionState.RUNNING,
            ],
        )

    def test_single_board_has_no_mimo_trigger_or_armed_wait(self):
        devices = _FakeMimoDevices(["master"], [11111])
        master = devices.get_device(0)
        self.assertFalse(master._mimo_master)
        self.assertEqual(master._trig_output_time, 0)
        devices.run()
        self.assertNotIn(("armed", "master"), _FakeDevice.calls)

    def test_inactive_slaves_use_direct_master_run(self):
        devices = _FakeMimoDevices(
            ["master", "unused"], [11111, 11111]
        )
        devices.get_device(1)._seq = None
        result = devices.run()
        self.assertFalse(devices.get_device(0)._mimo_master)
        self.assertNotIn(("armed", "unused"), _FakeDevice.calls)
        self.assertIn("warning", result[1][1])

    def test_slave_finishing_before_master_is_sync_failure(self):
        devices = _FakeMimoDevices(
            ["master", "early", "slow"], [11111, 11111, 11111]
        )
        devices.get_device(1).finish_before_master = True
        devices.get_device(2).armed_delay = 0.05
        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ):
            with self.assertRaisesRegex(
                RuntimeError, "completed before the master started"
            ):
                devices.run()
        self.assertNotIn(("master_started", "master"), _FakeDevice.calls)
        self.assertEqual(devices.session.state, AcquisitionState.FAILED)
        _FakeDevice.master_started.set()

    def test_armed_timeout_stops_every_registered_board_without_master_start(self):
        devices = _FakeMimoDevices(
            ["master", "slow"], [11111, 11111], armed_timeout=0.01
        )
        devices.get_device(1).armed_delay = 0.05
        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ) as stop_board:
            with self.assertRaisesRegex(TimeoutError, "waiting for MIMO slaves"):
                devices.run()

        self.assertNotIn(("master_started", "master"), _FakeDevice.calls)
        self.assertEqual(stop_board.call_count, 2)
        self.assertEqual(devices.session.state, AcquisitionState.FAILED)
        _FakeDevice.master_started.set()

    def test_all_compilation_finishes_before_any_board_is_registered(self):
        devices = _FakeMimoDevices(
            ["master", "bad"], [11111, 11111]
        )
        devices.get_device(1).prepare_error = RuntimeError("compile failed")
        with mock.patch.object(AcquisitionSession, "_stop_board") as stop_board:
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                devices.run()

        stop_board.assert_not_called()
        self.assertEqual(devices.session.participating_board_ids, ())
        self.assertFalse(any(call[0] == "send" for call in _FakeDevice.calls))


class DeviceProtocolTest(unittest.TestCase):
    @staticmethod
    def make_device(replies):
        device = Device.__new__(Device)
        device._seq_compiled = True
        device._s = _ReplySocket(replies)
        device._flush_old_rx = False
        device._machine_code = np.array([0x01000000], dtype=np.uint32)
        device._print_infos = False
        device._assert_errors = True
        device._grad_board = "ocra1_5761"
        device._rx0_cic_factor = 1
        device._rx1_cic_factor = 1
        device._close_socket = False
        return device

    @staticmethod
    def reply(run_seq):
        return [sc.reply_pkt, 1, 0, sc.version_full, {"run_seq": run_seq}, {}]

    def test_armed_intermediate_reply_is_not_treated_as_final(self):
        device = self.make_device([
            self.reply({"armed": True, "final": False}),
            self.reply({
                "final": True,
                "aborted": False,
                "rx0_i": [1],
                "rx0_q": [2],
            }),
        ])
        callbacks = []
        rxd, _ = device.run(
            report_armed=True,
            on_armed=lambda: callbacks.append("armed"),
            on_started=lambda: callbacks.append("started"),
            rearm=True,
            acquisition_id="scan-id",
        )
        request = msgpack.unpackb(device._s.sent[0])
        request_options = request[4]["run_seq"]
        self.assertEqual(callbacks, ["started", "armed"])
        self.assertTrue(request_options["report_armed"])
        self.assertTrue(request_options["rearm"])
        self.assertEqual(request_options["grad_board"], "ocra1_5761")
        self.assertEqual(request_options["acquisition_id"], "scan-id")
        self.assertEqual(len(rxd["rx0"]), 1)

    def test_aborted_final_reply_raises(self):
        device = self.make_device([
            self.reply({"final": True, "aborted": True}),
        ])
        with self.assertRaises(AcquisitionAborted):
            device.run(rearm=True)

    def test_cancelled_send_guard_prevents_rearm_packet(self):
        device = self.make_device([
            self.reply({"final": True, "aborted": False}),
        ])
        session = AcquisitionSession()
        session.begin()
        session.request_cancel()
        with self.assertRaises(AcquisitionAborted):
            device.run(rearm=True, send_context=session.send_guard)
        self.assertEqual(device._s.sent, [])

    def test_stop_does_not_wait_for_blocked_run_send(self):
        device = self.make_device([
            self.reply({"final": True, "aborted": True}),
        ])
        device._s = _BlockingReplySocket([
            self.reply({"final": True, "aborted": True}),
        ])
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.register_board(0, "master", 11111, "ocra1")

        run_errors = []
        run_thread = threading.Thread(
            target=lambda: self._run_and_capture_abort(device, session, run_errors)
        )
        run_thread.start()
        self.assertTrue(device._s.send_entered.wait(timeout=1))

        stop_done = threading.Event()
        stop_result = {}
        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ):
            stop_thread = threading.Thread(
                target=lambda: (
                    stop_result.update(session.stop()), stop_done.set()
                )
            )
            stop_thread.start()
            stopped_without_run_socket = stop_done.wait(timeout=0.2)
            device._s.release_send.set()
            stop_thread.join(timeout=1)

        run_thread.join(timeout=1)
        self.assertTrue(stopped_without_run_socket)
        self.assertEqual(stop_result, {0: "stopped"})
        self.assertEqual(len(run_errors), 1)
        self.assertIsInstance(run_errors[0], AcquisitionAborted)

    @staticmethod
    def _run_and_capture_abort(device, session, errors):
        try:
            device.run(
                rearm=True,
                send_context=session.send_guard,
                acquisition_id=session.acquisition_id,
            )
        except BaseException as exc:
            errors.append(exc)

    def test_emergency_reply_requires_all_safe_confirmations(self):
        status = {
            "ok": True,
            "latched": True,
            "fsm_idle": True,
            "digital_outputs_zeroed": True,
            "gradients_zeroed": True,
            "rx_fifo_empty": True,
        }
        reply = [
            sc.reply_pkt,
            1,
            0,
            sc.version_full,
            {"emergency_stop": status},
            {},
        ]
        self.assertIs(sc.validate_emergency_stop_reply(reply), reply)
        status["gradients_zeroed"] = False
        reply[5] = {"errors": ["Emergency stop safe-state check failed"]}
        with self.assertRaisesRegex(RuntimeError, "gradients_zeroed"):
            sc.validate_emergency_stop_reply(reply)
        status["gradients_zeroed"] = True
        reply[5] = {}
        reply[0] = 129
        with self.assertRaises(RuntimeError):
            sc.validate_emergency_stop_reply(reply)


def _set_event():
    event = threading.Event()
    event.set()
    return event


class _BoardSocket:
    def __init__(self, rig, board_id):
        self.rig, self.board_id = rig, board_id
        self.requests = []
        self.sent = threading.Event()
        self.armed = threading.Event()
        self.final_sent = threading.Event()
        self.closed = threading.Event()
        self.arm_gate = _set_event()
        self.finish_gate = _set_event()
        self.tail_points = 0
        self.final_error = None
        self.drop_final = False
        self.omit_armed = False
        self._buffer = b""

    def sendall(self, data):
        request = msgpack.unpackb(data, raw=False)[4]['run_seq']
        self.requests.append(request)
        self._frames = self.frames(request)
        self.sent.set()
        if self.board_id == 0:
            self.rig.armed_at_start = [wire.armed.is_set() for wire in self.rig.wires[1:]
                                       if self.rig.devices.dev(wire.board_id)._seq is not None]
            self.rig.master_started.set()

    def wait_for(self, event):
        deadline = time.monotonic() + 3
        while not event.wait(.005):
            if self.closed.is_set():
                raise ConnectionError('Fake board socket shut down')
            if time.monotonic() > deadline:
                raise TimeoutError('Test did not release the board')
        if self.closed.is_set():
            raise ConnectionError('Fake board socket shut down')

    @staticmethod
    def packet(data, error=None):
        reply = DeviceProtocolTest.reply(data)
        if error:
            reply[5] = {'errors': [error]}
        return msgpack.packb(reply)

    def frames(self, request):
        if request.get('report_armed'):
            self.wait_for(self.arm_gate)
            if self.omit_armed:
                yield self.packet({'final': True})
                return
            self.armed.set()
            yield self.packet({'armed': True, 'final': False})
        self.wait_for(self.rig.master_started)
        points = request.get('chunk_points', 0)
        channels = request.get('rx_channels', [0])

        def samples(count):
            return {f'rx{ch}_{part}': [ch + 1] * count
                    for ch in channels for part in ('i', 'q')}

        if points:
            yield self.packet(dict(samples(points), chunk_index=0, final=False))
        self.wait_for(self.finish_gate)
        if self.drop_final:
            return
        self.final_sent.set()
        tail = samples(self.tail_points) if points and self.tail_points else {}
        yield self.packet(dict(tail, chunk_index=1 if points else 0, final=True), self.final_error)

    def recv(self, size):
        if not self._buffer:
            self._buffer = next(self._frames, b'')
        # Exercise framing across arbitrary socket reads as well as concurrency.
        data, self._buffer = self._buffer[:17], self._buffer[17:]
        return data

    def shutdown(self, _how):
        self.closed.set()


class _Rig:
    def __init__(self, count):
        self.master_started = threading.Event()
        self.armed_at_start = []
        self.wires = [_BoardSocket(self, i) for i in range(count)]
        self.devices = MimoDevices.__new__(MimoDevices)
        self.devices._session = AcquisitionSession()
        self.devices._armed_timeout = .15
        self.devices._devs = []
        for i, wire in enumerate(self.wires):
            dev = DeviceProtocolTest.make_device([])
            dev._s = wire
            dev._seq = {'fake': ([], [])}
            dev._ip_address, dev._port = f'board{i}.invalid', 11111
            dev._mimo_master = i == 0 and count > 1
            # Real compilation/timing is covered by test_acquisition.RxGainTest.
            dev.disable_mimo_sync = lambda dev=dev: setattr(dev, '_mimo_master', False)
            self.devices._devs.append(dev)


class MimoStreamTest(unittest.TestCase):
    def setUp(self):
        self.rigs = []
        self.stop_ids = []
        self.enterContext(mock.patch.object(
            socket.socket, 'connect', side_effect=AssertionError('No real hardware in tests')))
        self.enterContext(mock.patch.object(
            AcquisitionSession, '_stop_board', side_effect=self.stop_board))

    def make_rig(self, count=2):
        self.rig = _Rig(count)
        self.rigs.append(self.rig)
        return self.rig

    def stop_board(self, board, timeout, acquisition_id):
        self.stop_ids.append(board.board_id)
        self.rig.wires[board.board_id].shutdown(socket.SHUT_RDWR)
        return {'ok': True}

    def tearDown(self):
        for rig in self.rigs:
            for wire in rig.wires:
                wire.shutdown(socket.SHUT_RDWR)
        self.assertFalse([thread.name for thread in threading.enumerate()
                          if thread.name.startswith('marga-board-')])

    def collect_async(self, stream):
        records, errors = [], []
        finished = threading.Event()

        def collect():
            try:
                records.extend(stream)
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=collect, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        return records, errors, finished

    def test_single_channel_on_master_or_any_slave_maps_back_to_logical_rx(self):
        for channel in range(6):
            with self.subTest(channel=channel):
                rig = self.make_rig(3)
                rig.wires[channel // 2].tail_points = 2
                starts = []
                records = list(rig.devices.run_stream(4, [channel], on_started=lambda: starts.append(True)))
                self.assertEqual(starts, [True])
                self.assertEqual(rig.armed_at_start, [True, True])
                self.assertEqual([item[2]['final'] for item in records], [False, True])
                self.assertEqual([len(item[0][f'rx{channel}']) for item in records], [4, 2])
                for data, _, _ in records:
                    self.assertEqual(set(data), {f'rx{channel}'})
                    np.testing.assert_allclose(data[f'rx{channel}'], (channel % 2 + 1) * (1 + 1j) / 2**24)
                for i, wire in enumerate(rig.wires):
                    request = wire.requests[0]
                    self.assertEqual(request['acquisition_id'], rig.devices.session.acquisition_id)
                    self.assertTrue(request['rearm'])
                    self.assertEqual(request['report_armed'], i != 0)
                    if i == channel // 2:
                        self.assertEqual(request['rx_channels'], [channel % 2])
                    else:
                        self.assertNotIn('chunk_points', request)
                self.assertEqual(rig.devices.session.state, AcquisitionState.RUNNING)

    def test_single_board_two_channel_stream_is_unchanged(self):
        rig = self.make_rig(1)
        records = list(rig.devices.run_stream(4, [0, 1]))
        self.assertEqual(set(records[0][0]), {'rx0', 'rx1'})
        self.assertEqual(records[-1][0], {})
        self.assertTrue(records[-1][2]['final'])
        self.assertFalse(rig.wires[0].requests[0]['report_armed'])

    def test_inactive_slaves_use_direct_master_stream(self):
        rig = self.make_rig(3)
        for board in rig.devices._devs[1:]:
            board._seq = None
        records = list(rig.devices.run_stream(4, [1]))
        self.assertIn('rx1', records[0][0])
        self.assertFalse(rig.devices.dev(0)._mimo_master)
        self.assertEqual(rig.devices.session.participating_board_ids, (0,))
        self.assertTrue(all(not wire.requests for wire in rig.wires[1:]))

    def test_chunks_arrive_while_master_runs_but_final_waits_for_master(self):
        rig = self.make_rig()
        rig.wires[0].finish_gate.clear()
        stream = rig.devices.run_stream(4, [3])
        first = next(stream)
        self.assertIn('rx3', first[0])
        self.assertFalse(rig.wires[0].final_sent.is_set())
        records, errors, finished = self.collect_async(stream)
        self.assertTrue(rig.wires[1].final_sent.wait(1))
        self.assertFalse(finished.wait(.05))
        rig.wires[0].finish_gate.set()
        self.assertTrue(finished.wait(1))
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], {})
        self.assertTrue(records[0][2]['final'])

    def test_late_master_error_is_not_hidden_by_rx_final(self):
        rig = self.make_rig()
        rig.wires[0].finish_gate.clear()
        rig.wires[0].final_error = 'master execution failed'
        records, errors, finished = self.collect_async(rig.devices.run_stream(4, [2]))
        self.assertTrue(rig.wires[1].final_sent.wait(1))
        self.assertFalse(finished.wait(.05))
        rig.wires[0].finish_gate.set()
        self.assertTrue(finished.wait(1))
        self.assertEqual(len(errors), 1)
        self.assertIn('master execution failed', str(errors[0]))
        self.assertFalse(any(record[2]['final'] for record in records))
        self.assertEqual(set(self.stop_ids), {0, 1})

    def test_stop_while_arming_never_starts_master(self):
        rig = self.make_rig()
        rig.wires[1].arm_gate.clear()
        _, errors, finished = self.collect_async(rig.devices.run_stream(4, [2]))
        self.assertTrue(rig.wires[1].sent.wait(1))
        rig.devices.session.stop()
        self.assertTrue(finished.wait(1))
        self.assertIsInstance(errors[0], AcquisitionAborted)
        self.assertEqual(rig.wires[0].requests, [])
        self.assertEqual(set(self.stop_ids), {0, 1})

    def test_stop_while_streaming_stops_all_boards(self):
        rig = self.make_rig()
        for wire in rig.wires:
            wire.finish_gate.clear()
        stream = rig.devices.run_stream(4, [2])
        next(stream)
        rig.devices.session.stop()
        with self.assertRaises(AcquisitionAborted):
            list(stream)
        self.assertEqual(set(self.stop_ids), {0, 1})
        self.assertEqual(rig.devices.session.state, AcquisitionState.ABORTED)

    def test_close_stream_early_stops_all_boards_and_workers(self):
        rig = self.make_rig()
        for wire in rig.wires:
            wire.finish_gate.clear()
        stream = rig.devices.run_stream(4, [2])
        next(stream)
        stream.close()
        self.assertEqual(set(self.stop_ids), {0, 1})
        self.assertTrue(all(wire.closed.is_set() for wire in rig.wires))

    def test_armed_timeout_stops_all_boards(self):
        rig = self.make_rig()
        rig.wires[1].arm_gate.clear()
        with self.assertRaisesRegex(TimeoutError, 'waiting for MIMO slaves'):
            list(rig.devices.run_stream(4, [2]))
        self.assertEqual(rig.wires[0].requests, [])
        self.assertEqual(set(self.stop_ids), {0, 1})

    def test_rx_eof_before_final_is_failure(self):
        rig = self.make_rig()
        rig.wires[1].drop_final = True
        with self.assertRaisesRegex(ConnectionError, 'before its final reply'):
            list(rig.devices.run_stream(4, [2]))
        self.assertEqual(rig.devices.session.state, AcquisitionState.FAILED)
        self.assertEqual(set(self.stop_ids), {0, 1})

    def test_missing_armed_is_failure_without_starting_master(self):
        rig = self.make_rig()
        rig.wires[1].omit_armed = True
        with self.assertRaisesRegex(RuntimeError, 'required armed event'):
            list(rig.devices.run_stream(4, [2]))
        self.assertEqual(rig.wires[0].requests, [])

    def test_invalid_or_multiple_channels_are_rejected_before_sending(self):
        for channels in (None, [], [0, 2], [-1], [4], [1.5]):
            with self.subTest(channels=channels):
                rig = self.make_rig()
                with self.assertRaises((ValueError, NotImplementedError)):
                    list(rig.devices.run_stream(4, channels))
                self.assertTrue(all(not wire.requests for wire in rig.wires))

    def test_missing_master_or_rx_sequence_is_rejected_before_sending(self):
        for missing in (0, 1):
            rig = self.make_rig()
            rig.devices.dev(missing)._seq = None
            with self.assertRaisesRegex(ValueError, 'both have sequence data'):
                list(rig.devices.run_stream(4, [2]))
            self.assertEqual(rig.devices.session.participating_board_ids, ())
        self.assertEqual(self.stop_ids, [])

    def test_compile_failure_does_not_register_or_start_boards(self):
        rig = self.make_rig()
        with mock.patch.object(rig.devices.dev(1), 'prepare', side_effect=RuntimeError('compile failed')):
            with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                list(rig.devices.run_stream(4, [2]))
        self.assertEqual(rig.devices.session.participating_board_ids, ())
        self.assertEqual(self.stop_ids, [])
        self.assertTrue(all(not wire.requests for wire in rig.wires))

    def test_repeated_scans_reuse_session_and_rearm_slaves(self):
        rig = self.make_rig()
        for _ in range(2):
            rig.master_started.clear()
            for wire in rig.wires:
                wire.armed.clear()
            records = list(rig.devices.run_stream(4, [2]))
            self.assertEqual(rig.armed_at_start, [True])
            self.assertEqual(len(records), 2)
        self.assertEqual(len(rig.wires[1].requests), 2)
        self.assertEqual({request['acquisition_id'] for request in rig.wires[1].requests},
                         {rig.devices.session.acquisition_id})


class StreamArmedProtocolTest(unittest.TestCase):
    def test_armed_is_reported_once_and_not_exposed_as_chunk(self):
        packet = DeviceProtocolTest.reply
        device = DeviceProtocolTest.make_device([
            packet({'armed': True, 'final': False}),
            packet({'armed': True, 'final': False}),
            packet({'rx1_i': [1], 'rx1_q': [2], 'chunk_index': 0, 'final': True}),
        ])
        armed = []
        records = list(device.run_stream(4, [1], report_armed=True, on_armed=lambda: armed.append(True)))
        self.assertEqual(armed, [True])
        self.assertEqual(len(records), 1)
        self.assertEqual(set(records[0][0]), {'rx1'})
        request = msgpack.unpackb(device._s.sent[0], raw=False)[4]['run_seq']
        self.assertTrue(request['report_armed'])

    def test_zero_chunk_size_preserves_armed_handshake(self):
        packet = DeviceProtocolTest.reply
        device = DeviceProtocolTest.make_device([
            packet({'armed': True, 'final': False}),
            packet({'rx0_i': [1], 'rx0_q': [2], 'final': True}),
        ])
        armed = []
        records = list(device.run_stream(0, [0], report_armed=True, on_armed=lambda: armed.append(True)))
        self.assertEqual(armed, [True])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0][2]['final'])


if __name__ == "__main__":
    unittest.main()
