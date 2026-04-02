import yaml
from pathlib import Path

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / "config.yaml"

with open(_CONFIG_PATH, "r") as _f:
    cfg = yaml.safe_load(_f)
