#!/usr/bin/env python3

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
        self.assertEqual(session.stop(), {})
        self.assertEqual(session.state, AcquisitionState.ABORTED)
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
        ):
            self.assertEqual(session.stop(), {0: "stopped"})
        self.assertEqual(session.state, AcquisitionState.IDLE)

    def test_terminal_stop_returns_to_idle(self):
        session = AcquisitionSession()
        session.begin()
        session.finish()
        session.stop()
        self.assertEqual(session.state, AcquisitionState.IDLE)

    def test_second_stop_after_confirmed_abort_returns_to_idle(self):
        session = AcquisitionSession()
        session.begin()
        session.transition(AcquisitionState.STARTING_MASTER)
        session.register_board(0, "master", 11111, "ocra1")
        with mock.patch.object(
            AcquisitionSession, "_stop_board", return_value="stopped"
        ):
            self.assertEqual(session.stop(), {0: "stopped"})

        self.assertEqual(session.state, AcquisitionState.ABORTED)
        self.assertEqual(session.stop(), {})
        self.assertEqual(session.state, AcquisitionState.IDLE)

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


if __name__ == "__main__":
    unittest.main()
