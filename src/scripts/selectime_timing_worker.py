"""Internal fresh-process worker; experiment options belong to the Hydra timing entry."""
import json
from pathlib import Path
import sys

from timebench.pipeline.timing import cold_sample
from timebench.pipeline.workflow import write_json


if __name__ == '__main__':
    request, result = map(Path, sys.argv[1:])
    write_json(result, cold_sample(json.loads(request.read_text(encoding='utf-8'))))
