#!/usr/bin/env bash
# ===================================================================
#  TEIA - Reconhecimento de Padroes
#  INSTALADOR (macOS / Linux) - equivalente ao dependencias.bat.
#
#  Uso:  ./dependencias.sh
# ===================================================================
set -u
cd "$(dirname "$0")" || exit 1

echo
echo " ==================================================================="
echo "   TEIA - Reconhecimento de Padroes"
echo "   Instalador de dependencias"
echo " ==================================================================="
echo

# ------------------------------------------------------------ 1. Python
echo " [1/3] Procurando o Python..."

PY=""
if [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYCMD="python"
else
    echo
    echo "   Python nao encontrado."
    echo "   Instale com o gerenciador do seu sistema, por exemplo:"
    echo "     macOS         brew install python"
    echo "     Ubuntu/Debian sudo apt install python3 python3-venv python3-pip"
    echo
    exit 1
fi

# ------------------------------------------------------------ 2. venv
if [ -z "$PY" ]; then
    "$PYCMD" --version
    echo
    echo " [2/3] Criando o ambiente do projeto..."
    if ! "$PYCMD" -m venv venv; then
        echo
        echo "   Nao consegui criar o ambiente virtual."
        echo "   No Ubuntu/Debian pode faltar o pacote: sudo apt install python3-venv"
        echo
        exit 1
    fi
    PY="venv/bin/python"
fi
echo "       Ambiente pronto  (pasta venv)"

# ------------------------------------------------------- 3. bibliotecas
echo
echo " [3/3] Instalando as bibliotecas do projeto..."
echo
"$PY" -m pip install --upgrade pip --disable-pip-version-check --quiet
if ! "$PY" -m pip install -r requirements.txt --disable-pip-version-check; then
    echo
    echo "   A instalacao das bibliotecas falhou. Verifique a internet e"
    echo "   leia a mensagem de erro acima."
    echo
    exit 1
fi

echo
echo "       Conferindo a instalacao..."
if ! "$PY" -c "import fastapi, uvicorn, xlrd, matplotlib, sklearn" 2>/dev/null; then
    echo "   Alguma biblioteca nao foi instalada corretamente."
    exit 1
fi
echo "       Tudo certo."

# ------------------------------------------- 4. interface ja compilada
echo
echo " [extra] Conferindo a interface..."
if [ -f "web_app/frontend/dist/index.html" ]; then
    echo "       Interface ja compilada - o Node.js nao e necessario."
else
    echo "       A interface ainda nao esta compilada."
    if command -v npm >/dev/null 2>&1; then
        echo "       Node.js encontrado - compilando agora (demora um pouco)..."
        npm --prefix web_app/frontend install
        npm --prefix web_app/frontend run build
        echo "       Interface compilada."
    else
        echo
        echo "   Instale o Node.js 20+ (https://nodejs.org) e rode este arquivo"
        echo "   de novo, ou copie a pasta web_app/frontend/dist ja compilada."
        echo
        exit 1
    fi
fi

echo
echo " ==================================================================="
echo "   PRONTO! Para abrir o projeto:   ./iniciar.sh"
echo " ==================================================================="
echo
