"""Bounded cancellation of owned process groups; no process-name matching."""
from __future__ import annotations
import contextlib
import os
import signal
import subprocess
import threading
import time


@contextlib.contextmanager
def cleaning_up():
    """A second Ctrl+C must not interrupt child reaping or lock release."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def install_signals():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)


def stop(proc, grace=2):
    """Only processes started with start_new_session=True may be passed here."""
    if type(proc.pid) is not int or proc.pid <= 1 or proc.pid == os.getpgrp():
        raise ValueError("Refusing to signal an invalid or caller-owned process group")
    def send(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    with cleaning_up():
        send(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Also catch descendants left behind after their group leader exits.
            send(signal.SIGKILL)
            proc.wait()


def communicate(proc, input=None, *, timeout=None, cancel=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if cancel is not None and cancel.is_set():
            raise KeyboardInterrupt
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        try:
            return proc.communicate(input=input, timeout=min(0.2, remaining) if remaining is not None else 0.2)
        except subprocess.TimeoutExpired:
            input = None  # communicate retains unsent input across timeout retries.


def run(args, *, input=None, check=False, timeout=None, cancel=None, keep_stdin=False,
        grace=2, capture_output=False, **kwargs):
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if input is not None:
        kwargs['stdin'] = subprocess.PIPE
    else:
        kwargs.setdefault('stdin', subprocess.DEVNULL)
    if keep_stdin:
        kwargs['bufsize'] = 0  # Closing the liveness pipe must never flush buffered request data.
    proc = subprocess.Popen(args, start_new_session=True, **kwargs)
    connection = None
    try:
        if keep_stdin:
            # Send one JSON line but retain the pipe as a parent-liveness channel.
            # communicate normally closes stdin; detach it until the operation completes.
            connection, proc.stdin = proc.stdin, None
            connection.write(input)
            connection.flush()
            input = None
        stdout, stderr = communicate(proc, input, timeout=timeout, cancel=cancel)
        result = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    except BaseException:
        with cleaning_up():
            if connection is not None:
                try:
                    connection.close()
                except BrokenPipeError:
                    pass
                connection = None
                # Give the agent time to reap its children and acknowledge EOF before
                # closing SSH. If the connection is hung, escalation remains bounded.
                try:
                    communicate(proc, timeout=grace)
                except subprocess.TimeoutExpired:
                    stop(proc, grace=0)
            else:
                stop(proc, grace)
        raise
    finally:
        if connection is not None:
            connection.close()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def pipeline(producer, consumer):
    writer = subprocess.Popen(producer, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, start_new_session=True)
    pipe = writer.stdout
    reader = None
    try:
        reader = subprocess.Popen(consumer, stdin=writer.stdout, start_new_session=True)
        pipe.close()
        writer.stdout = None
        communicate(reader)
        if reader.returncode:
            raise subprocess.CalledProcessError(reader.returncode, consumer)
        communicate(writer)
        if writer.returncode:
            raise subprocess.CalledProcessError(writer.returncode, producer)
    except BaseException:
        with cleaning_up():
            if reader is not None:
                stop(reader)
            stop(writer)
        raise
    finally:
        pipe.close()
