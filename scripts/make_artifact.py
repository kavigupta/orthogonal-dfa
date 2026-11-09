"""Writes the paper's anonymized artifact from the committed files at HEAD.

python scripts/make_artifact.py artifact.zip
"""

import io
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = (
    "orthogonal_dfa/",
    "data/",
    "figures/",
    "proofs/",
    "tests/",
    "scripts/run_all_experiments.sh",
    "scripts/capal_run_all.sh",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "LICENSE",
)
# Unused by the paper, and they name our lab's machines.
EXCLUDE = (
    "data/pretrained_models/fm-1.traced.pt",
    "orthogonal_dfa/spliceai/evaluate_lssi.py",
    "orthogonal_dfa/spliceai/best_psams_for_lssi.py",
)
REMOVE = {
    "setup.py": [
        '    author_email="orthogonaldfa@kavigupta.org",\n',
        '    url="https://github.com/kavigupta/orthogonal-dfa",\n',
        "    # documentation\n"
        "    project_urls={\n"
        '        "Documentation": "https://orthogonal-dfa.readthedocs.io/en/latest/#",\n'
        "    },\n",
    ],
    "setup.cfg": ["author = Kavi Gupta\n"],
}
IDENTIFYING = re.compile(
    rb"(?i)kavi|gupta|csail|mit\.edu|expeditions|/scratch/|/mnt/|/home/"
)


def committed_files():
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isfile():
                yield member.name, member.mode, tar.extractfile(member).read()


def contents(path, data):
    """``data``, and each member's bytes if it is itself a zip, as checkpoints are."""
    yield path, data
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                yield f"{path}:{name}", z.read(name)


def main():
    (out,) = sys.argv[1:]
    files = {}
    for path, mode, data in committed_files():
        if path.startswith(INCLUDE) and path not in EXCLUDE:
            files[path] = mode, data
    _, readme = files["orthogonal_dfa/experiments/README.md"]
    files["README.md"] = 0o644, readme
    for path, removals in REMOVE.items():
        mode, data = files[path]
        text = data.decode()
        for removal in removals:
            assert removal in text, (path, removal)
            text = text.replace(removal, "")
        files[path] = mode, text.encode()

    found = [
        (inner, m.group().decode(errors="replace"))
        for path, (_, data) in files.items()
        for inner, part in contents(path, data)
        for m in IDENTIFYING.finditer(part)
    ]
    assert not found, found

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, (mode, data) in sorted(files.items()):
            info = zipfile.ZipInfo(f"artifact/{path}", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    print(f"Wrote {out}: {len(files)} files")


if __name__ == "__main__":
    main()
