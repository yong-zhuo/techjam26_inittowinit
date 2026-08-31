from __future__ import annotations

import os
import pathlib
import shutil
import stat
import sys

# Regenerates submission/ from starter/. The two must never be edited independently.
ROOT = pathlib.Path(__file__).resolve().parent
SRC, DST = ROOT / "starter", ROOT / "submission"
BUNDLED = ("requirements.txt", ".env.example", "demo_session.py")

# dotted and slashed paths only; prose such as "the starter agent provided" is left alone
RULES = (
    ("starter.src", "submission.src"),
    ("starter.agent", "submission.agent"),
    ("starter/assets", "submission/assets"),
    ("starter/src", "submission/src"),
    ("Participant starter package", "Participant submission package"),
)
TEXT = {".py", ".md", ".txt", ".example"}


def _force(func, path, _exc):
    # read-only or briefly locked files, common on Windows with __pycache__
    os.chmod(path, stat.S_IWRITE)
    func(path)


def build() -> list[pathlib.Path]:
    if DST.exists():
        assets = DST / "assets"
        if assets.is_dir() and assets.is_symlink():
            assets.unlink()
        shutil.rmtree(DST, onexc=_force)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("assets", "__pycache__", "*.pyc"))
    for name in BUNDLED:
        shutil.copy2(ROOT / name, DST / name)

    rewritten = []
    for path in sorted(DST.rglob("*")):
        if path.is_dir() or path.suffix not in TEXT:
            continue
        text = original = path.read_text(encoding="utf-8")
        for old, new in RULES:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            rewritten.append(path.relative_to(DST))
    return rewritten


def verify() -> list[str]:
    problems = []
    for path in sorted(DST.rglob("*")):
        if path.is_dir() or path.suffix not in TEXT or "assets" in path.parts:
            continue
        rel = path.relative_to(DST)
        twin = SRC / rel
        if not twin.exists():
            if rel.name not in BUNDLED:
                problems.append(f"only in submission: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in RULES:
            text = text.replace(new, old)
        if text != twin.read_text(encoding="utf-8"):
            problems.append(f"differs after reversing the rename: {rel}")
    return problems


def main() -> None:
    rewritten = build()
    problems = verify()
    print(f"submission/ rebuilt from starter/, {len(rewritten)} files rewritten")
    if problems:
        print("MISMATCH:")
        for item in problems:
            print(f"  {item}")
        sys.exit(1)
    print("verified: every shared file is identical after reversing the rename")


if __name__ == "__main__":
    main()
