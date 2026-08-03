"""Minimal in-tree PEP 517 build backend for the gainz_kernels package.

Why this exists: the trusted runner installs the participant kernel package
with ``pip install --no-deps /submission/Sources/kernels`` from a
**read-only** bind mount. The default setuptools path fails there — its
``egg_info``/``build`` steps write into the source tree ("could not create
'gainz_kernels.egg-info': Read-only file system", observed in the
gainz-candidate container logs). This backend builds the wheel entirely in
pip's temporary directories using only the standard library:

- ``requires = []`` in ``[build-system]`` → the isolated build env needs no
  downloads, so the install also works with no network egress.
- No file is ever written under the source tree; the wheel zip is
  assembled in memory and written to pip's ``wheel_directory``.

The wheel contains the ``gainz_kernels`` package plus dist-info metadata
declaring the ``vllm.general_plugins`` entry point so vLLM calls
``gainz_kernels.register()`` at engine startup.
"""

import base64
import hashlib
import os
import zipfile

NAME = "gainz_kernels"
DIST_NAME = "gainz-kernels"
VERSION = "0.5.0"
TAG = "py3-none-any"

_METADATA = (
    "Metadata-Version: 2.1\n"
    f"Name: {DIST_NAME}\n"
    f"Version: {VERSION}\n"
    "Summary: Participant CUDA/Triton kernels loaded into the candidate "
    "vLLM engine by the trusted gainz.fast runner.\n"
    "Requires-Python: >=3.10\n"
)

_WHEEL = (
    "Wheel-Version: 1.0\n"
    "Generator: gainz-inline-backend (0.5.0)\n"
    "Root-Is-Purelib: true\n"
    f"Tag: {TAG}\n"
)

_ENTRY_POINTS = "[vllm.general_plugins]\ngainz = gainz_kernels:register\n"

# Fixed timestamp keeps the wheel byte-reproducible across builds.
_ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def _package_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, os.pardir, NAME))


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _collect_files() -> list[tuple[str, bytes]]:
    src = _package_dir()
    entries: list[tuple[str, bytes]] = []
    for root, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(names):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(root, name)
            arcname = NAME + "/" + os.path.relpath(path, src).replace(os.sep, "/")
            with open(path, "rb") as handle:
                entries.append((arcname, handle.read()))
    return entries


def get_requires_for_build_wheel(config_settings=None):  # noqa: ARG001
    return []


def get_requires_for_build_sdist(config_settings=None):  # noqa: ARG001
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: ARG001
    dist_info = f"{NAME}-{VERSION}.dist-info"
    files = _collect_files()
    files.append((f"{dist_info}/METADATA", _METADATA.encode("utf-8")))
    files.append((f"{dist_info}/WHEEL", _WHEEL.encode("utf-8")))
    files.append((f"{dist_info}/entry_points.txt", _ENTRY_POINTS.encode("utf-8")))

    record_lines = [
        f"{arcname},{_record_hash(data)},{len(data)}" for arcname, data in files
    ]
    record_lines.append(f"{dist_info}/RECORD,,")
    record = ("\n".join(record_lines) + "\n").encode("utf-8")

    wheel_name = f"{NAME}-{VERSION}-{TAG}.whl"
    wheel_path = os.path.join(wheel_directory, wheel_name)
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, data in files + [(f"{dist_info}/RECORD", record)]:
            info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return wheel_name


def build_sdist(sdist_directory, config_settings=None):  # noqa: ARG001
    import io
    import tarfile

    base = f"{NAME}-{VERSION}"
    sdist_name = f"{base}.tar.gz"
    sdist_path = os.path.join(sdist_directory, sdist_name)
    root = os.path.normpath(os.path.join(_package_dir(), os.pardir))
    with tarfile.open(sdist_path, "w:gz") as archive:
        for rel in ["pyproject.toml", f"_build/gainz_backend.py"]:
            path = os.path.join(root, rel)
            if os.path.exists(path):
                archive.add(path, arcname=f"{base}/{rel}")
        for arcname, data in _collect_files():
            info = tarfile.TarInfo(f"{base}/{arcname}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return sdist_name
