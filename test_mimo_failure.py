"""Offline regressions for failures while another MIMO board is still running."""
from contextlib import contextmanager
import threading
import unittest
from unittest import mock

from acquisition import AcquisitionAborted, AcquisitionSession, AcquisitionState
from mimo_devices import MimoDevices
from test_acquisition import _FakeDevice


class _ControlledDevice(_FakeDevice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.error = None

    def run(self, report_armed=False, on_armed=None, on_started=None,
            send_context=None, **kwargs):
        try:
            with send_context():
                if on_started is not None:
                    on_started()
            if report_armed:
                on_armed()
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("Test did not release board reply")
            if self.error is not None:
                raise self.error
            return ({"rx0": [self._ip_address]}, {})
        finally:
            self.exited.set()


class _ControlledMimo(MimoDevices):
    device_class = _ControlledDevice


class MimoFailureTest(unittest.TestCase):
    @contextmanager
    def running(self):
        with mock.patch("socket.socket", side_effect=AssertionError("Hardware I/O forbidden")), \
                mock.patch.object(AcquisitionSession, "_stop_board", return_value="stopped") as stop:
            devices = _ControlledMimo(["master", "slave", "unused"], [11111] * 3)
            devices.dev(2)._seq = None
            done = threading.Event()
            results, errors = [], []

            def run():
                try:
                    results.append(devices.run())
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    done.set()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                self.assertTrue(devices.dev(0).entered.wait(2))
                self.assertTrue(devices.dev(1).entered.wait(2))
                yield devices, done, results, errors, stop
            finally:
                for dev in devices.dev_list():
                    dev.release.set()
                thread.join(3)
                for dev in devices.dev_list()[:2]:
                    self.assertTrue(dev.exited.wait(3))
                self.assertFalse(thread.is_alive())

    def test_slave_failure_stops_all_boards_before_master_returns(self):
        cases = (
            (AcquisitionAborted("slave aborted"), AcquisitionState.ABORTED),
            (ConnectionError("slave disconnected"), AcquisitionState.FAILED),
            # An error already returned by the server, not FIFO-full detection.
            (AssertionError("RX FIFO/s full during sequence"), AcquisitionState.FAILED),
        )
        for error, state in cases:
            with self.subTest(error=type(error).__name__), self.running() as run:
                devices, done, results, errors, stop = run
                devices.dev(1).error = error
                devices.dev(1).release.set()
                self.assertTrue(done.wait(1), "Slave failure waited for the master reply")
                self.assertFalse(devices.dev(0).exited.is_set())
                self.assertEqual(errors, [error])
                self.assertEqual(results, [])
                self.assertEqual(devices.session.state, state)
                self.assertCountEqual([c.args[0].board_id for c in stop.call_args_list], [0, 1])

    def test_master_failure_stops_slaves_and_preserves_original_error(self):
        with self.running() as run:
            devices, done, results, errors, stop = run
            error = ConnectionError("master disconnected")
            devices.dev(0).error = error
            devices.dev(0).release.set()
            self.assertTrue(done.wait(1))
            self.assertFalse(devices.dev(1).exited.is_set())
            self.assertEqual(errors, [error])
            self.assertEqual(results, [])
            self.assertEqual(devices.session.state, AcquisitionState.FAILED)
            self.assertEqual(stop.call_count, 2)

    def test_success_waits_for_both_boards_and_preserves_result_order(self):
        for first in (0, 1):
            with self.subTest(first=first), self.running() as run:
                devices, done, results, errors, stop = run
                devices.dev(first).release.set()
                self.assertTrue(devices.dev(first).exited.wait(1))
                self.assertFalse(done.wait(.1))
                devices.dev(1 - first).release.set()
                self.assertTrue(done.wait(1))
                self.assertEqual(errors, [])
                self.assertEqual([result[0] for result in results[0]],
                                 [{"rx0": ["master"]}, {"rx0": ["slave"]}, {}])
                self.assertEqual(devices.session.state, AcquisitionState.RUNNING)
                stop.assert_not_called()

    def test_stop_during_blocked_replies_cancels_coordinator(self):
        with self.running() as run:
            devices, done, results, errors, stop = run
            devices.emergency_stop()
            self.assertTrue(done.wait(1))
            self.assertEqual(results, [])
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], AcquisitionAborted)
            self.assertEqual(devices.session.state, AcquisitionState.ABORTED)
            self.assertEqual(stop.call_count, 2)


if __name__ == "__main__":
    unittest.main()
