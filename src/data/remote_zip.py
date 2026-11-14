"""Read single members of a large remote zip over HTTP range requests (no full download).

Used to fetch a handful of Sugar Beets 2016 photos out of 1.6 GB archives:
    zipfile.ZipFile(HttpFile(url)).read(member)
"""
from __future__ import annotations

import requests


class HttpFile:
    """Minimal read-only, seekable file over HTTP (the server must support `Range` requests)."""

    def __init__(self, url: str, timeout: float = 60.0):
        self.url, self.pos, self.timeout = url, 0, timeout
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        head.raise_for_status()
        if head.headers.get("Accept-Ranges") != "bytes":
            raise OSError(f"{url} does not support range requests")
        self.size = int(head.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        response = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=self.timeout)
        response.raise_for_status()
        self.pos += len(response.content)
        return response.content
