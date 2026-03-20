import subprocess
import logging

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 20
MAX_LINE_LENGTH = 200


def _truncate_lines(text: str) -> str:
    lines = text.splitlines()

    if len(lines) > MAX_LOG_LINES:
        head = lines[:10]
        tail = lines[-10:]
        lines = head + ["... (truncated) ..."] + tail

    truncated = []
    for line in lines:
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "..."
        truncated.append(line)

    return "\n".join(truncated)


def run_command(cmd: list[str], cwd=None):
    logger.info("Running: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if proc.stdout:
        logger.debug("stdout:\n%s", _truncate_lines(proc.stdout))

    if proc.stderr:
        logger.warning("stderr:\n%s", _truncate_lines(proc.stderr))

    if proc.returncode != 0:
        logger.error("Command failed with exit code %d", proc.returncode)
        raise RuntimeError("External command failed")

    return proc