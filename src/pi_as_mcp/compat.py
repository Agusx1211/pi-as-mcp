from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pi_as_mcp import __version__

# Bump this when the request envelope or daemon compatibility response changes
# incompatibly. BUILD_ID catches code skew within the same protocol revision.
DAEMON_PROTOCOL_VERSION = 1
COMPAT_COMMAND = "__compat__"
CHECKED_REQUEST_COMMAND = "__checked_request__"


def _source_build_id() -> str:
    package_dir = Path(__file__).parent
    digest = hashlib.sha256()
    digest.update(
        f"pi-as-mcp\0{__version__}\0daemon-protocol\0{DAEMON_PROTOCOL_VERSION}\0".encode()
    )
    for source in sorted(package_dir.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(b"\0")
        try:
            digest.update(source.read_bytes())
        except OSError:
            # Normal wheels and editable installs include sources. Retaining the
            # filename still gives source-less repackagers a stable identity.
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()[:24]


DAEMON_BUILD_ID = _source_build_id()


def compatibility_identity() -> dict[str, Any]:
    return {
        "protocol_version": DAEMON_PROTOCOL_VERSION,
        "build_id": DAEMON_BUILD_ID,
        "package_version": __version__,
    }
