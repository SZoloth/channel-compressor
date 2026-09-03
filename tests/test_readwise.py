from channel_compressor.readwise import (
    ReaderClient,
    _merge_channel_compressor_note,
    _reader_tag_names,
)


class _FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.responses = [
            _FakeResponse(429, headers={"Retry-After": "0.2"}),
            _FakeResponse(200, payload={"ok": True}),
        ]
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_reader_retries_rate_limit(monkeypatch):
    slept = []
    client = ReaderClient(token="test-token")
    client.session = _FakeSession()
    monkeypatch.setattr("channel_compressor.readwise.time.sleep", lambda value: slept.append(value))

    response = client._request("GET", "https://example.test")

    assert response.json() == {"ok": True}
    assert client.session.calls == 2
    assert slept == [0.2]


def test_reader_sync_helpers_preserve_user_state():
    assert _reader_tag_names([{"name": "existing"}, "plain"]) == {"existing", "plain"}
    first = _merge_channel_compressor_note("My own note", "Rank 1")
    second = _merge_channel_compressor_note(first, "Rank 2")
    assert second.startswith("My own note")
    assert second.count("[Channel Compressor selection]") == 1
    assert second.endswith("Rank 2")
