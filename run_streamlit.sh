#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] Streamlit 전용 가상환경을 만듭니다."
  python3 -m venv .venv
fi

source .venv/bin/activate
if [ ! -f ".venv/.dashboard_ready" ]; then
  echo "[2/3] 최초 실행용 패키지를 설치합니다."
  python -m pip install --upgrade pip
  python -m pip install -r requirements-dashboard.txt
  touch .venv/.dashboard_ready
else
  echo "[2/3] 설치된 환경을 재사용합니다."
fi

echo "[3/3] 대시보드를 실행합니다."
python -m streamlit run dashboard_app.py
