#!/usr/bin/env python3
"""Shared acquisition lifecycle and emergency-stop coordination.

The session is deliberately independent from the GUI.  Sequence builders can
use :meth:`checkpoint` while preparing locally, and ``MimoDevices`` registers
boards immediately before hardware participation becomes possible.  This lets
the same Stop operation cover both phases without sharing a blocking
``run_seq`` socket with the emergency request.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import socket
import threading
import uuid
import warnings

import server_comms as sc


class AcquisitionState(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    ARMING_SLAVES = "ARMING_SLAVES"
    WAITING_ALL_ARMED = "WAITING_ALL_ARMED"
    STARTING_MASTER = "STARTING_MASTER"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


class AcquisitionAborted(RuntimeError):
    """Raised at a cooperative checkpoint after Stop has been requested."""


@dataclass(frozen=True)
class _Board:
    board_id: int
    ip_address: str
    port: int
    grad_board: str = None


_TERMINAL_STATES = {
    AcquisitionState.FINISHED,
    AcquisitionState.ABORTED,
    AcquisitionState.FAILED,
}

_ALLOWED_TRANSITIONS = {
    AcquisitionState.IDLE: {AcquisitionState.PREPARING},
    AcquisitionState.PREPARING: {
        AcquisitionState.ARMING_SLAVES,
        AcquisitionState.STARTING_MASTER,
        AcquisitionState.FINISHED,
        AcquisitionState.ABORTED,
        AcquisitionState.FAILED,
    },
    AcquisitionState.ARMING_SLAVES: {
        AcquisitionState.WAITING_ALL_ARMED,
        AcquisitionState.ABORTED,
        AcquisitionState.FAILED,
    },
    AcquisitionState.WAITING_ALL_ARMED: {
        AcquisitionState.STARTING_MASTER,
        AcquisitionState.ABORTED,
        AcquisitionState.FAILED,
    },
    AcquisitionState.STARTING_MASTER: {
        AcquisitionState.RUNNING,
        AcquisitionState.ABORTED,
        AcquisitionState.FAILED,
    },
    AcquisitionState.RUNNING: {
        # A sequenceRun may contain multiple batches/scans.  RUNNING is the
        # stable state between hardware runs; finalisation belongs to the
        # higher-level sequence lifecycle.
        AcquisitionState.ARMING_SLAVES,
        AcquisitionState.STARTING_MASTER,
        AcquisitionState.FINISHED,
        AcquisitionState.ABORTED,
        AcquisitionState.FAILED,
    },
    AcquisitionState.FINISHED: {
        AcquisitionState.IDLE,
        AcquisitionState.PREPARING,
    },
    AcquisitionState.ABORTED: {
        AcquisitionState.IDLE,
        AcquisitionState.PREPARING,
    },
    AcquisitionState.FAILED: {
        AcquisitionState.IDLE,
        AcquisitionState.PREPARING,
    },
}


class AcquisitionSession:
    """Thread-safe state and cancellation owner for one acquisition session."""

    def __init__(self, on_state_change=None):
        self._lock = threading.RLock()
        self._stop_lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._state = AcquisitionState.IDLE
        self._boards = {}
        self._on_state_change = on_state_change
        self._last_stop_results = {}
        self._stop_dispatched = False
        self._acquisition_id = None

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def cancelled(self):
        return self._cancel_event.is_set()

    @property
    def participating_board_ids(self):
        with self._lock:
            return tuple(sorted(self._boards))

    @property
    def last_stop_results(self):
        with self._lock:
            return dict(self._last_stop_results)

    @property
    def stop_failures(self):
        """Return board-indexed failures from the latest emergency dispatch."""
        with self._lock:
            return {
                board_id: str(result)
                for board_id, result in self._last_stop_results.items()
                if isinstance(result, BaseException)
            }

    @property
    def acquisition_id(self):
        with self._lock:
            return self._acquisition_id

    def set_state_callback(self, callback):
        with self._lock:
            self._on_state_change = callback

    def _notify(self, state):
        with self._lock:
            callback = self._on_state_change
        if callback is not None:
            try:
                callback(state)
            except Exception as exc:  # lifecycle safety must not depend on UI code
                warnings.warn(
                    f"Acquisition state callback failed in {state.value}: {exc}",
                    RuntimeWarning,
                )

    def transition(self, state):
        state = AcquisitionState(state)
        with self._lock:
            current = self._state
            if current == state:
                return state
            if self._cancel_event.is_set() and state not in _TERMINAL_STATES:
                raise AcquisitionAborted("Acquisition stop requested")
            if state not in _ALLOWED_TRANSITIONS[current]:
                raise RuntimeError(
                    f"Invalid acquisition state transition: {current.value} -> {state.value}"
                )
            self._state = state
        self._notify(state)
        return state

    def begin(self):
        """Explicitly begin/rearm a new local acquisition lifecycle."""
        # Do not let a new ID start while emergency packets for the previous ID
        # are still in flight.
        with self._stop_lock:
            with self._lock:
                current = self._state
                if current not in ({AcquisitionState.IDLE} | _TERMINAL_STATES):
                    raise RuntimeError(
                        f"Cannot begin acquisition while state is {current.value}"
                    )
                unresolved_hardware_stop = (
                    current in {AcquisitionState.ABORTED, AcquisitionState.FAILED}
                    and self._boards
                    and (
                        not self._stop_dispatched
                        or any(
                            isinstance(result, BaseException)
                            for result in self._last_stop_results.values()
                        )
                    )
                )
                if unresolved_hardware_stop:
                    raise RuntimeError(
                        "Cannot begin a new acquisition before every participating "
                        "board confirms its emergency-safe state"
                    )
                self._cancel_event.clear()
                self._boards.clear()
                self._last_stop_results = {}
                self._stop_dispatched = False
                self._acquisition_id = str(uuid.uuid4())
                self._state = AcquisitionState.PREPARING
        self._notify(AcquisitionState.PREPARING)
        return AcquisitionState.PREPARING

    def checkpoint(self):
        if self._cancel_event.is_set():
            raise AcquisitionAborted("Acquisition stop requested")

    @contextmanager
    def send_guard(self):
        """Check cancellation at the send boundary without blocking Stop I/O.

        Boards are registered before this point.  Once the check succeeds, a
        racing Stop may immediately use its independent connection; a run
        packet that reaches the server afterwards is rejected by the latched
        ``acquisition_id``.  In particular, the session lock must not be held
        across a large or blocked ``socket.sendall()``.
        """
        with self._lock:
            self.checkpoint()
        yield

    def request_cancel(self):
        """Synchronously latch cancellation without performing network I/O."""
        with self._lock:
            current = self._state
            if current in _TERMINAL_STATES:
                return current
            if current == AcquisitionState.IDLE:
                return current
            self._cancel_event.set()
            self._state = AcquisitionState.ABORTED
            self._stop_dispatched = False
        self._notify(AcquisitionState.ABORTED)
        return AcquisitionState.ABORTED

    def register_board(self, board_id, ip_address, port, grad_board=None):
        """Register a board before any request that might make it participate."""
        board = _Board(
            int(board_id),
            str(ip_address),
            int(port),
            None if grad_board is None else str(grad_board),
        )
        with self._lock:
            self.checkpoint()
            self._boards[board.board_id] = board

    @staticmethod
    def _stop_board(board, timeout, acquisition_id):
        # A fresh socket is mandatory: the normal Device socket may be blocked
        # waiting for the final run_seq response.
        with socket.create_connection(
            (board.ip_address, board.port), timeout=timeout
        ) as stop_socket:
            stop_socket.settimeout(timeout)
            payload = {}
            if board.grad_board is not None:
                payload["grad_board"] = board.grad_board
            if acquisition_id is not None:
                payload["acquisition_id"] = acquisition_id
            packet = sc.construct_packet(payload, command=sc.emergency_stop_pkt)
            reply = sc.send_packet(packet, stop_socket)
            return sc.validate_emergency_stop_reply(reply)

    def _stop_participating_boards(self, timeout):
        with self._lock:
            boards = tuple(self._boards.values())
            acquisition_id = self._acquisition_id

        results = {}
        if boards:
            with ThreadPoolExecutor(max_workers=len(boards)) as executor:
                pending = {
                    executor.submit(
                        self._stop_board, board, timeout, acquisition_id
                    ): board
                    for board in boards
                }
                for future in as_completed(pending):
                    board = pending[future]
                    try:
                        results[board.board_id] = future.result()
                    except Exception as exc:
                        results[board.board_id] = exc

        failures = {
            board_id: str(result)
            for board_id, result in results.items()
            if isinstance(result, BaseException)
        }
        notify_failed = False
        with self._lock:
            self._last_stop_results = dict(results)
            if failures and self._state != AcquisitionState.FAILED:
                self._state = AcquisitionState.FAILED
                notify_failed = True
        if notify_failed:
            self._notify(AcquisitionState.FAILED)
        return results

    def ensure_hardware_stopped(self, timeout=2.0):
        """Idempotently abort and dispatch emergency stops if not already done."""
        with self._stop_lock:
            notify_aborted = False
            with self._lock:
                if self._stop_dispatched:
                    return dict(self._last_stop_results)
                self._cancel_event.set()
                if self._state not in _TERMINAL_STATES:
                    self._state = AcquisitionState.ABORTED
                    notify_aborted = True
                self._stop_dispatched = True
            if notify_aborted:
                self._notify(AcquisitionState.ABORTED)
            return self._stop_participating_boards(timeout)

    def stop(self, timeout=2.0):
        """Request Stop and concurrently emergency-stop every participating board.

        In ``PREPARING`` there are no registered boards, so this is a purely
        local cooperative cancellation.  Aborted/failed sessions stay latched
        until an explicit ``begin()``; repeated Stop must not let unfinished
        work interpret IDLE as a new acquisition.  FINISHED may return to IDLE.
        """
        # A second concurrently arriving Stop waits for the first dispatch and
        # returns its result.  It must not interpret ABORTED as a later terminal
        # reset and clear the first dispatch's board snapshot.
        if not self._stop_lock.acquire(blocking=False):
            with self._stop_lock:
                return self.last_stop_results

        try:
            notify_aborted = False
            with self._lock:
                current = self._state
                if current == AcquisitionState.ABORTED:
                    if self._stop_dispatched:
                        return dict(self._last_stop_results)
                    else:
                        reset_terminal = False
                        self._stop_dispatched = True
                elif current == AcquisitionState.FAILED:
                    self._cancel_event.set()
                    needs_retry = bool(self._boards) and (
                        not self._stop_dispatched
                        or any(
                            isinstance(result, BaseException)
                            for result in self._last_stop_results.values()
                        )
                    )
                    if not needs_retry:
                        return dict(self._last_stop_results)
                    reset_terminal = False
                    self._stop_dispatched = True
                elif current == AcquisitionState.FINISHED:
                    reset_terminal = True
                elif current == AcquisitionState.IDLE:
                    return {}
                else:
                    reset_terminal = False
                    self._cancel_event.set()
                    self._state = AcquisitionState.ABORTED
                    self._stop_dispatched = True
                    notify_aborted = True

            if reset_terminal:
                self.reset()
                return {}

            if notify_aborted:
                self._notify(AcquisitionState.ABORTED)
            return self._stop_participating_boards(timeout)
        finally:
            self._stop_lock.release()

    def finish(self):
        self.checkpoint()
        return self.transition(AcquisitionState.FINISHED)

    def abort(self):
        with self._lock:
            self._cancel_event.set()
            if self._state == AcquisitionState.ABORTED:
                return AcquisitionState.ABORTED
            if self._state in _TERMINAL_STATES:
                return self._state
            self._state = AcquisitionState.ABORTED
        self._notify(AcquisitionState.ABORTED)
        return AcquisitionState.ABORTED

    def fail(self, stop_hardware=False, timeout=2.0):
        with self._stop_lock:
            if stop_hardware:
                self._cancel_event.set()
                with self._lock:
                    dispatch_stop = not self._stop_dispatched
                    self._stop_dispatched = True
                if dispatch_stop:
                    self._stop_participating_boards(timeout)
            with self._lock:
                if self._state == AcquisitionState.FAILED:
                    return AcquisitionState.FAILED
                if self._state in _TERMINAL_STATES:
                    return self._state
                self._state = AcquisitionState.FAILED
        self._notify(AcquisitionState.FAILED)
        return AcquisitionState.FAILED

    def reset(self):
        with self._stop_lock:
            with self._lock:
                if self._state == AcquisitionState.IDLE:
                    return AcquisitionState.IDLE
                if self._state not in _TERMINAL_STATES:
                    raise RuntimeError(
                        f"Cannot reset acquisition while state is {self._state.value}"
                    )
                self._state = AcquisitionState.IDLE
                self._boards.clear()
                self._last_stop_results = {}
                self._stop_dispatched = False
                self._acquisition_id = None
                # Keep cancellation latched until begin() explicitly starts a
                # new acquisition; late work cannot escape a Stop.
        self._notify(AcquisitionState.IDLE)
        return AcquisitionState.IDLE

    def snapshot(self):
        with self._lock:
            return {
                "state": self._state.value,
                "cancelled": self._cancel_event.is_set(),
                "participating_board_ids": tuple(sorted(self._boards)),
                "stop_dispatched": self._stop_dispatched,
                "stop_failures": {
                    board_id: str(result)
                    for board_id, result in self._last_stop_results.items()
                    if isinstance(result, BaseException)
                },
                "acquisition_id": self._acquisition_id,
            }
