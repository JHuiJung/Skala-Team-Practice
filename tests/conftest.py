"""pytest가 프로젝트 루트를 기준으로 `src` 패키지를 임포트할 수 있도록 경로를 추가한다."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
