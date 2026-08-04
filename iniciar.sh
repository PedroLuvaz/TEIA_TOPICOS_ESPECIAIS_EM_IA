#!/usr/bin/env bash
# ===================================================================
#  TEIA - Reconhecimento de Padroes
#  Sobe o projeto e abre o navegador (macOS / Linux).
# ===================================================================
cd "$(dirname "$0")" || exit 1

PY="./venv/bin/python"
[ -x "$PY" ] || PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"

exec "$PY" iniciar.py "$@"
