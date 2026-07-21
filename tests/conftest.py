"""pytest가 프로젝트 루트를 기준으로 `src` 패키지를 임포트할 수 있도록 경로를 추가한다."""

import sys
from pathlib import Path

import matplotlib

# 테스트는 화면 없는 환경(CI 등)에서도 돌아가야 하므로, 다른 모듈이 pyplot을
# 임포트하기 전에 렌더링 전용 백엔드(Agg)로 고정한다.
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
