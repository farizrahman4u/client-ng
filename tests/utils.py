import requests
import json
import os
from tests.mock_server import create_app
from _pytest.config import get_config
from pytest_mock import _get_mock_module


def subdict(d, expected_dict):
    """Return a new dict with only the items from `d` whose keys occur in `expected_dict`.
    """
    return {k: v for k, v in d.items() if k in expected_dict}


def fixture_open(path):
    """Returns an opened fixture file"""
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fixtures", path))


def default_ctx():
    return {
        "fail_count": 0,
        "page_count": 0,
        "page_times": 2,
        "files": {},
    }


def mock_server():
    ctx = default_ctx()
    app = create_app(ctx)
    mock = RequestsMock(app, ctx)
    mocker = _get_mock_module(get_config())
    # We mock out all requests libraries, couldn't find a way to mock the core lib
    mocker.patch("gql.transport.requests.requests", mock).start()
    mocker.patch("wandb.internal.file_stream.requests", mock).start()
    mocker.patch("wandb.internal.internal_api.requests", mock).start()
    mocker.patch("wandb.internal.update.requests", mock).start()
    mocker.patch("wandb.apis.internal_runqueue.requests", mock).start()
    mocker.patch("wandb.apis.public.requests", mock).start()
    mocker.patch("wandb.util.requests", mock).start()
    mocker.patch("wandb.wandb_sdk.wandb_artifacts.requests", mock).start()
    print("Patched requests everywhere", os.getpid())
    return mock


class ResponseMock(object):
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.response.status_code >= 400:
            raise requests.exceptions.HTTPError("Bad Request", response=self.response)

    @property
    def content(self):
        return self.response.data.decode('utf-8')

    def iter_content(self, chunk_size=1024):
        yield self.response.data

    def json(self):
        return json.loads(self.response.data.decode('utf-8'))


class RequestsMock(object):
    def __init__(self, app, ctx):
        self.app = app
        self.client = app.test_client()
        self.ctx = ctx

    def set_context(self, key, value):
        self.ctx[key] = value

    def Session(self):
        return self

    @property
    def RequestException(self):
        return requests.RequestException

    @property
    def HTTPError(self):
        return requests.HTTPError

    @property
    def headers(self):
        return {}

    @property
    def utils(self):
        return requests.utils

    @property
    def exceptions(self):
        return requests.exceptions

    @property
    def packages(self):
        return requests.packages

    @property
    def adapters(self):
        return requests.adapters

    def mount(self, *args):
        pass

    def _clean_kwargs(self, kwargs):
        if "auth" in kwargs:
            del kwargs["auth"]
        if "timeout" in kwargs:
            del kwargs["timeout"]
        if "cookies" in kwargs:
            del kwargs["cookies"]
        if "params" in kwargs:
            del kwargs["params"]
        if "stream" in kwargs:
            del kwargs["stream"]
        if "verify" in kwargs:
            del kwargs["verify"]
        if "allow_redirects" in kwargs:
            del kwargs["allow_redirects"]
        return kwargs

    def _store_request(self, url, body):
        key = url.split("/")[-1]
        self.ctx[key] = self.ctx.get(key, [])
        self.ctx[key].append(body)

    def post(self, url, **kwargs):
        self._store_request(url, kwargs.get("json"))
        return ResponseMock(self.client.post(url, **self._clean_kwargs(kwargs)))

    def put(self, url, **kwargs):
        self._store_request(url, kwargs.get("json"))
        return ResponseMock(self.client.put(url, **self._clean_kwargs(kwargs)))

    def get(self, url, **kwargs):
        self._store_request(url, kwargs.get("json"))
        return ResponseMock(self.client.get(url, **self._clean_kwargs(kwargs)))

    def request(self, method, url, **kwargs):
        if method.lower() == "get":
            self.get(url, **kwargs)
        elif method.lower() == "post":
            self.post(url, **kwargs)
        elif method.lower() == "put":
            self.put(url, **kwargs)
        else:
            message = "Request method not implemented: %s" % method
            raise requests.RequestException(message)

    def __repr__(self):
        return "<W&B Mocked Request class>"
