import types
import builtins
from types import SimpleNamespace

import pytest

from andrio_toolbox import UE5IntegratedTools

class DummyConnection:
    def __init__(self, *args, **kwargs):
        self.opened = False

    def open_connection(self):
        self.opened = True

    def close_connection(self):
        self.opened = False

    def execute_python_command(self, command, *args, **kwargs):
        return SimpleNamespace(output_str=f'executed:{command}')

    def flush(self):
        pass


def test_connect_success(monkeypatch):
    def dummy_init(cfg):
        return DummyConnection()
    monkeypatch.setattr('upyrc.upyre.PythonRemoteConnection', dummy_init)
    tools = UE5IntegratedTools()
    assert tools.connect() is True
    assert tools.is_connected()
    out = tools.execute_python('1+1')
    assert 'executed:1+1' == out


def test_connect_failure(monkeypatch):
    class FailConn(DummyConnection):
        def open_connection(self):
            raise RuntimeError('no ue')
    monkeypatch.setattr('upyrc.upyre.PythonRemoteConnection', lambda cfg: FailConn())
    tools = UE5IntegratedTools()
    assert tools.connect() is False
    assert not tools.is_connected()


def test_execute_without_connection():
    tools = UE5IntegratedTools()
    assert tools.execute_python('print(123)') == '❌ Not connected to UE5 remote execution'

