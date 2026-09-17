"""Per-process proxy setup for public model/data preparation; no credential I/O."""
import os
from urllib.parse import urlsplit


def configure_download_proxy():
    proxy = os.environ.get('ROBOCURATE_HF_PROXY') or os.environ.get('HOLOCURATE_HF_PROXY')
    if proxy:
        if urlsplit(proxy).scheme not in ('http', 'https'):
            raise ValueError('ROBOCURATE_HF_PROXY must be an HTTP(S) proxy URL')
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            os.environ[key] = proxy
        for key in ('ALL_PROXY', 'all_proxy'):
            os.environ.pop(key, None)
    else:
        specific = [os.environ.get(k, '') for k in ('HTTPS_PROXY', 'https_proxy')]
        for key in ('ALL_PROXY', 'all_proxy'):
            if urlsplit(os.environ.get(key, '')).scheme == 'socks':
                if any(urlsplit(p).scheme in ('http', 'https') for p in specific):
                    # HTTPX parses ALL_PROXY even when an HTTPS proxy is set.
                    os.environ.pop(key, None)
                else:
                    raise ValueError('Unsupported socks:// proxy. Set ROBOCURATE_HF_PROXY to your valid HTTP proxy URL.')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
