import sys
from path import Path
import importlib
import pkg_resources

PROJECT_ROOT = Path(__file__).parent
SRC: Path = PROJECT_ROOT / 'src'


def update_pythonpath():
    sys.path += list(SRC.walkdirs())
    sys.path.append(PROJECT_ROOT / 'costants')
    sys.path.append('src/')


def check_requirements():
    with open(PROJECT_ROOT / 'requirements.txt') as f:
        pkg_resources.require(f.read().splitlines())


def run_module():
    try:
        cmd = sys.argv[1]
        module = importlib.import_module(cmd)
        module.main()
    except:
        raise AttributeError("You must specify the input argument")


update_pythonpath()
check_requirements()
run_module()