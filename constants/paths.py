from path import Path

PRJ_ROOT = Path(__file__).parent.parent
SRC = PRJ_ROOT / 'src'
DATA = PRJ_ROOT / 'data'
DATA_RAW = DATA / 'raw'
DATA_PROCESSING = DATA / 'processing'
DATA_PROCESSED = DATA / 'processed'
MODELS = PRJ_ROOT / 'models'
NOTEBOOKS = PRJ_ROOT / 'notebooks'
REPORT = PRJ_ROOT / 'report'
OUTPUT = PRJ_ROOT / 'output'
for path in [SRC, DATA, DATA_RAW, DATA_PROCESSING, DATA_PROCESSED, MODELS, NOTEBOOKS, REPORT, OUTPUT]:
    path.mkdir_p()
SCORE_FILE : Path = OUTPUT / 'scores.json'
