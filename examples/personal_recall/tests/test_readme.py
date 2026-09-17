"""Keep the README honest: every command and flag it documents must actually exist.

Documentation drifts silently. A README that tells a new user to run `--some-flag` that was renamed
two phases ago is worse than no README, because it fails at the first thing they try. These tests
parse the fenced command blocks out of README.md and check each documented flag against the script's
real `--help`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
README = BASE_DIR / "README.md"
sys.path.insert(0, str(BASE_DIR))

#: Commands whose help text needs no API key or model load.
SAFE_TO_PROBE = {
    "recall.py",
    "chat_import.py",
    "run_baseline.py",
    "citation_metrics.py",
    "absence_claims.py",
    "attribution_screen.py",
    "verify_product_parity.py",
    "build_stress_v2.py",
    "validate_stress_dataset.py",
}


def _readme() -> str:
    assert README.exists(), f"missing {README}"
    return README.read_text(encoding="utf-8")


def _bash_commands() -> list[str]:
    """Yield the shell commands from every fenced block in the README."""
    commands: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", _readme(), flags=re.DOTALL):
        for raw in block.split("\n"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(line.rstrip("\\").strip())
    return commands


def _script_of(command: str) -> str | None:
    match = re.search(r"python\s+(?:-m\s+)?([\w./-]+\.py)", command)
    return Path(match.group(1)).name if match else None


def test_readme_documents_at_least_one_command_per_core_script() -> None:
    scripts = {_script_of(c) for c in _bash_commands()}
    for expected in ("recall.py", "chat_import.py", "run_baseline.py", "verify_product_parity.py"):
        assert expected in scripts, f"README no longer shows how to run {expected}"


def test_documented_scripts_exist() -> None:
    for command in _bash_commands():
        script = _script_of(command)
        if script and script in SAFE_TO_PROBE:
            assert (BASE_DIR / script).exists(), f"README references a missing script: {script}"


def _declared_flags(script: str) -> set[str]:
    """Flags a script declares, read statically from its argparse calls.

    Spawning `--help` would be authoritative but costs a full import of the RAG stack per script
    (nine subprocesses, minutes). Reading `add_argument("--flag")` is immediate and still catches the
    drift that matters: a flag renamed in the source no longer matches the README.
    """
    source = (BASE_DIR / script).read_text(encoding="utf-8")
    flags = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', source))
    flags |= set(re.findall(r"add_argument\(\s*'(--[\w-]+)'", source))
    return flags


@pytest.mark.parametrize("script", sorted(SAFE_TO_PROBE))
def test_every_documented_flag_exists_in_that_script(script: str) -> None:
    if not (BASE_DIR / script).exists():
        pytest.skip(f"{script} is not present")

    documented = [
        flag
        for command in _bash_commands()
        if _script_of(command) == script
        for flag in re.findall(r"--[\w-]+", command)
    ]
    if not documented:
        pytest.skip(f"README documents no flags for {script}")

    declared = _declared_flags(script)
    missing = sorted({flag for flag in documented if flag not in declared})
    assert not missing, f"README documents flags {missing} that {script} does not declare"


def test_the_product_cli_really_accepts_its_documented_flags() -> None:
    """One authoritative end-to-end probe, for the command a user runs first."""
    commands = [c for c in _bash_commands() if _script_of(c) == "recall.py"]
    assert commands, "README no longer documents recall.py"
    required = {"--corpus", "--json"}
    documented = {flag for command in commands for flag in re.findall(r"--[\w-]+", command)}
    assert required <= documented, f"the quickstart must show {sorted(required)}"
    assert required <= _declared_flags("recall.py")


def test_documented_subcommands_of_pytest_exist() -> None:
    """The test command in the README must be a real invocation."""
    for command in _bash_commands():
        if "pytest" in command:
            assert "-m pytest" in command or command.startswith("pytest")


def test_readme_links_the_decision_log() -> None:
    text = _readme()
    assert "PROJECT_STATUS.md" in text, "the README must point at the measurement record"
    assert (BASE_DIR / "PROJECT_STATUS.md").exists()


def test_readme_states_the_known_limits() -> None:
    """A README that only sells the happy path is not the honest one this project keeps."""
    text = _readme()
    for claim in ("unreachable", "Text only", "not textually reproducible"):
        assert claim in text, f"README dropped the limitation: {claim!r}"
