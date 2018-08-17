# ============ Base imports ======================
import subprocess as sp
# ====== External package imports ================
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def syscall_decode(commands=list(), stdout=sp.PIPE, stderr=sp.PIPE, timeout=None, check=True, bufsize=-1):
    logger.debug("Running subprocess:{}".format(commands))
    run = sp.run(commands, stdout=stdout, stderr=stderr, timeout=timeout, check=check, bufsize=bufsize)
    stdout = run.stdout.decode('UTF-8') if run.stdout is not None else None
    stderr = run.stderr.decode('UTF-8') if run.stderr is not None else None
    return stdout, stderr, run.returncode


def syscall(commands=list(), stdout=sp.PIPE, stderr=sp.PIPE, timeout=None, bufsize=-1):
    logger.debug("Running subprocess:{}".format(commands))
    p = sp.Popen(commands, stdout=stdout, stderr=stderr, bufsize=bufsize)
    results = p.communicate(timeout=timeout)
    return results[0], results[1], p.returncode


def stdin_syscall_decode(commands=list(), stdin="", bufsize=-1):
    p = sp.Popen(commands, stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
    results = p.communicate(input=stdin)
    stdout = results[0].decode('UTF-8') if results[0] is not None else None
    stderr = results[0].decode('UTF-8') if results[1] is not None else None
    return stdout, stderr, p.returncode


def stdin_syscall(commands=list(), stdin="", bufsize=-1):
    p = sp.Popen(commands, stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
    results = p.communicate(input=stdin)
    return results[0], results[1], p.returncode
