#!/usr/bin/env python3

import msgpack, warnings
from contextlib import nullcontext

from marmachine import MarServerWarning

version_major = 1
version_minor = 0
version_debug = 7
version_full = (version_major << 16) | (version_minor << 8) | version_debug

request_pkt = 0
emergency_stop_pkt = 1
close_server_pkt = 2
reply_pkt = 128

_EMERGENCY_SAFE_FIELDS = (
    "ok",
    "latched",
    "fsm_idle",
    "digital_outputs_zeroed",
    "gradients_zeroed",
    "rx_fifo_empty",
)


def construct_packet(
    data,
    packet_idx=0,
    command=request_pkt,
    version=(version_major, version_minor, version_debug),
):
    vma, vmi, vd = version
    assert vma < 256 and vmi < 256 and vd < 256, "Version is too high for a byte!"
    version = (vma << 16) | (vmi << 8) | vd
    fields = [command, packet_idx, 0, version, data]
    return fields


def send_packet(packet, socket):
    socket.sendall(msgpack.packb(packet))

    unpacker = msgpack.Unpacker()
    packet_done = False
    while not packet_done:
        buf = socket.recv(1024)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker:  # ugly way of doing it
            return o  # quit function after 1st reply (could make this a thread in the future)


def send_packet_stream(packet, socket, on_sent=None, send_context=None):
    """Send a request and yield successive replies until ``run_seq.final``.

    This is used both for chunk data and for the MIMO slave ``armed`` event.
    Every reply is a complete msgpack message carrying a ``run_seq`` map.  The
    optional callback runs immediately after the bytes have been sent, which
    lets the acquisition state machine distinguish STARTING from RUNNING.
    """
    context = send_context() if send_context is not None else nullcontext()
    with context:
        socket.sendall(msgpack.packb(packet))
        if on_sent is not None:
            on_sent()

    unpacker = msgpack.Unpacker()
    while True:
        buf = socket.recv(1 << 20)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker:
            yield o
            # o is [reply_type, reply_index, 0, version, data_map, messages_map]
            data_map = o[4] if len(o) > 4 else {}
            run_seq = data_map.get("run_seq") if isinstance(data_map, dict) else None
            if isinstance(run_seq, dict) and run_seq.get("final"):
                return


def process_reply(reply, print_infos=False, assert_errors=False):
    """Process the standard server messages attached to a reply."""
    if reply is None:
        raise ConnectionError("Server closed the connection without a reply")
    return_status = reply[5]

    if print_infos and return_status.get("infos"):
        print("Server info:")
        for k in return_status["infos"]:
            print(k)

    if return_status.get("warnings"):
        for k in return_status["warnings"]:
            warnings.warn(k, MarServerWarning)

    if return_status.get("errors"):
        if assert_errors:
            raise AssertionError(return_status["errors"][0])
        else:
            for k in return_status["errors"]:
                warnings.warn("SERVER ERROR: " + k, RuntimeWarning)

    return reply, return_status


def validate_emergency_stop_reply(reply):
    """Require the server to positively attest every emergency-safe result."""
    if reply is None:
        raise ConnectionError("Server closed the emergency-stop connection without a reply")
    if not isinstance(reply, (list, tuple)) or len(reply) < 6 or reply[0] != reply_pkt:
        raise RuntimeError("Server returned an invalid emergency-stop reply")
    messages = reply[5] if len(reply) > 5 and isinstance(reply[5], dict) else {}
    data = reply[4] if len(reply) > 4 and isinstance(reply[4], dict) else {}
    status = data.get("emergency_stop")
    if not isinstance(status, dict):
        raise RuntimeError("Server did not return emergency-stop safety status")
    unsafe = [field for field in _EMERGENCY_SAFE_FIELDS if not status.get(field)]
    if unsafe:
        detail = "Emergency stop did not confirm safe state: " + ", ".join(unsafe)
        if messages.get("errors"):
            detail += "; " + "; ".join(str(error) for error in messages["errors"])
        raise RuntimeError(detail)
    if messages.get("errors"):
        raise RuntimeError("; ".join(str(error) for error in messages["errors"]))
    return reply


def command(server_dict, socket, print_infos=False, assert_errors=False):
    packet = construct_packet(server_dict)
    reply = send_packet(packet, socket)
    return process_reply(reply, print_infos, assert_errors)
