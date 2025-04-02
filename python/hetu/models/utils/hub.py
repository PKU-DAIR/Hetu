from urllib.parse import urlparse

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

__all__ = ["is_remote_url"]