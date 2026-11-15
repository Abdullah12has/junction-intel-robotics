# Guarded rerun of perception_timing.py: hardware and socket guards and stubs are installed BEFORE any import.
import builtins
import os
import runpy
import socket
import subprocess
import sys
import types

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'
LOG = E + '/logs/guard_hits.txt'
BLOCK = ('/dev/ttyACM', '/dev/video', '/dev/tty', '/dev/serial', '/dev/ttyUSB', '/dev/media')


def hit(what):
    with open(LOG, 'a') as f:
        f.write(what + '\n')
    raise OSError('E1 guard blocked: ' + what)


def blocked_path(p):
    try:
        s = os.fspath(p)
    except TypeError:
        return False
    if isinstance(s, bytes):
        s = s.decode('utf-8', 'replace')
    return s.startswith(BLOCK)


_open = builtins.open
_os_open = os.open


def guarded_open(file, *a, **k):
    if blocked_path(file):
        hit('open ' + str(file))
    return _open(file, *a, **k)


def guarded_os_open(path, *a, **k):
    if blocked_path(path):
        hit('os.open ' + str(path))
    return _os_open(path, *a, **k)


builtins.open = guarded_open
os.open = guarded_os_open


def no_connect(self, *a, **k):
    hit('socket.connect')


def no_connect_ex(self, *a, **k):
    hit('socket.connect_ex')


def no_create_connection(*a, **k):
    hit('socket.create_connection')


socket.socket.connect = no_connect
socket.socket.connect_ex = no_connect_ex
socket.create_connection = no_create_connection


class NoPopen(object):
    def __init__(self, *a, **k):
        hit('subprocess.Popen')


subprocess.Popen = NoPopen
os.system = lambda *a, **k: hit('os.system')

import cv2  # noqa: E402


class NoCapture(object):
    def __init__(self, *a, **k):
        hit('cv2.VideoCapture')


cv2.VideoCapture = NoCapture


def stub(name):
    m = types.ModuleType(name)

    def missing(attr):
        def f(*a, **k):
            hit(name + '.' + attr)
        return f
    m.__getattr__ = missing
    sys.modules[name] = m


for _name in ('serial', 'pyrealsense2'):
    stub(_name)

sys.argv = [E + '/tools/perception_timing.py', E + '/perception']
runpy.run_path(E + '/tools/perception_timing.py', run_name='__main__')
