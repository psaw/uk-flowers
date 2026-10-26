"""Run independent checks and retain their real exit codes and logs."""

import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "pilot"
    report_dir.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) == 2 else "all"
    if mode not in {"all", "smoke", "quality"}:
        raise SystemExit("Expected all, smoke or quality")
    python = sys.executable
    checks = []
    if mode in {"all", "smoke"}:
        checks.append(
            (
                "smoke",
                [python, "-m", "pytest", "tests/pilot", "-q", "--junitxml=reports/pilot/smoke.xml"],
            )
        )
        checks.append(
            (
                "build",
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--no-build-isolation",
                    "--out-dir",
                    "reports/pilot/wheel",
                ],
            )
        )
    if mode in {"all", "quality"}:
        checks.extend(
            [
                ("lint", [python, "-m", "ruff", "check", "--no-fix", "src", "tests"]),
                ("format", [python, "-m", "ruff", "format", "--check", "src", "tests"]),
                ("types", [python, "-m", "mypy", "src", "tests"]),
                ("security", [python, "-m", "bandit", "-r", "-c", "pyproject.toml", "src"]),
                (
                    "tests",
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests",
                        "--cov=src",
                        "--cov-fail-under=80",
                        "--cov-report=term-missing",
                        "--cov-report=xml:reports/pilot/coverage.xml",
                        "--junitxml=reports/pilot/tests.xml",
                    ],
                ),
            ]
        )
    results = []
    for name, command in checks:
        print(f"Running {name}", flush=True)
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
            output, returncode = result.stdout, result.returncode
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or b""
            output = captured.decode(errors="replace") if isinstance(captured, bytes) else captured
            output += "\nCheck exceeded 1200 seconds.\n"
            returncode = 124
        (report_dir / f"{name}.log").write_text(output, encoding="utf-8")
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        results.append(
            {"check": name, "exit_code": returncode, "seconds": round(time.monotonic() - start, 2)}
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()
    )
    summary = {"commit": commit, "working_tree_dirty": dirty, "mode": mode, "checks": results}
    (report_dir / f"summary-{mode}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(item["exit_code"] != 0 for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
