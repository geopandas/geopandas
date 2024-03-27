"""Vendored, cut down version of pyogrio/util.py for use with fiona"""

import re
import sys
from urllib.parse import urlparse


def vsi_path(path: str) -> str:
    """
    Ensure path is a local path or a GDAL-compatible vsi path.

    """

    # path is already in GDAL format
    if path.startswith("/vsi"):
        return path

    # Windows drive letters (e.g. "C:\") confuse `urlparse` as they look like
    # URL schemes
    if sys.platform == "win32" and re.match("^[a-zA-Z]\\:", path):
        if not path.split("!")[0].endswith(".zip"):
            return path

        # prefix then allow to proceed with remaining parsing
        path = f"zip://{path}"

    path, archive, scheme = _parse_uri(path)

    if scheme or archive or path.endswith(".zip"):
        return _construct_vsi_path(path, archive, scheme)

    return path


# Supported URI schemes and their mapping to GDAL's VSI suffix.
SCHEMES = {
    "file": "file",
    "zip": "zip",
    "tar": "tar",
    "gzip": "gzip",
    "http": "curl",
    "https": "curl",
    "ftp": "curl",
    "s3": "s3",
    "gs": "gs",
    "az": "az",
    "adls": "adls",
    "adl": "adls",  # fsspec uses this
    "hdfs": "hdfs",
    "webhdfs": "webhdfs",
    # GDAL additionally supports oss and swift for remote filesystems, but
    # those are for now not added as supported URI
}

CURLSCHEMES = {k for k, v in SCHEMES.items() if v == "curl"}


def _parse_uri(path: str):
    """
    Parse a URI

    Returns a tuples of (path, archive, scheme)

    path : str
        Parsed path. Includes the hostname and query string in the case
        of a URI.
    archive : str
        Parsed archive path.
    scheme : str
        URI scheme such as "https" or "zip+s3".
    """
    parts = urlparse(path)

    # if the scheme is not one of GDAL's supported schemes, return raw path
    if parts.scheme and not all(p in SCHEMES for p in parts.scheme.split("+")):
        return path, "", ""

    # we have a URI
    path = parts.path
    scheme = parts.scheme or ""

    if parts.query:
        path += "?" + parts.query

    if parts.scheme and parts.netloc:
        path = parts.netloc + path

    parts = path.split("!")
    path = parts.pop() if parts else ""
    archive = parts.pop() if parts else ""
    return (path, archive, scheme)


def _construct_vsi_path(path, archive, scheme) -> str:
    """Convert a parsed path to a GDAL VSI path"""

    prefix = ""
    suffix = ""
    schemes = scheme.split("+")

    if "zip" not in schemes and (archive.endswith(".zip") or path.endswith(".zip")):
        schemes.insert(0, "zip")

    if schemes:
        prefix = "/".join(
            "vsi{0}".format(SCHEMES[p]) for p in schemes if p and p != "file"
        )

        if schemes[-1] in CURLSCHEMES:
            suffix = f"{schemes[-1]}://"

    if prefix:
        if archive:
            return "/{}/{}{}/{}".format(prefix, suffix, archive, path.lstrip("/"))
        else:
            return "/{}/{}{}".format(prefix, suffix, path)

    return path
