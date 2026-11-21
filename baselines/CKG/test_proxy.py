"""Offline tests for the baseline's network proxy defaults."""
import importlib.util
from pathlib import Path


def load_agent_module():
    path = Path(__file__).with_name("agent.py")
    spec = importlib.util.spec_from_file_location("test_baseline_agent_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_network_proxy_sets_all_variants(monkeypatch):
    agent = load_agent_module()
    names = (
        "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
        "ALL_PROXY", "all_proxy", "FTP_PROXY", "ftp_proxy",
        "NO_PROXY", "no_proxy",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    agent.configure_network_proxy()
    assert agent.os.environ["HTTP_PROXY"] == "http://10.70.254.202:11321"
    assert agent.os.environ["https_proxy"] == "http://10.70.254.202:11321"
    assert agent.os.environ["NO_PROXY"] == "localhost,127.0.0.1,::1"
