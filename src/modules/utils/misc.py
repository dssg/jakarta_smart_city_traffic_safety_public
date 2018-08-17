# ============ Base imports ======================
import traceback
import sys
# ====== External package imports ================
# ====== Internal package imports ================
# ============== Logging  ========================
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def run_and_catch_exceptions(logger, f, *args, **kwargs):
    try:
        result = f(*args, **kwargs)
    except Exception as e:
        exc_info = sys.exc_info()
        logger.error("".join(traceback.format_exception(*exc_info)))
        raise e
    return result