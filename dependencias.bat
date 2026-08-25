@echo off
REM ===================================================================
REM  TEIA - Reconhecimento de Padroes
REM  INSTALADOR: prepara tudo o que o projeto precisa para funcionar.
REM  Duplo clique aqui UMA VEZ. Depois, use "Iniciar Projeto.bat".
REM
REM  Sem acentos de proposito: consoles antigos do Windows trocam os
REM  caracteres acentuados por simbolos estranhos.
REM ===================================================================
title TEIA - Instalando as dependencias
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo  ===================================================================
echo    TEIA - Reconhecimento de Padroes
echo    Instalador de dependencias
echo  ===================================================================
echo.
echo    Este programa vai:
echo      1. procurar o Python no computador
echo      2. criar a pasta "venv", um Python so para este projeto
echo      3. baixar as bibliotecas que o projeto usa
echo.
echo    Precisa de internet. Na primeira vez leva alguns minutos.
echo.
pause

REM ------------------------------------------------------------ 1. Python
echo.
echo  [1/3] Procurando o Python...

if exist "venv\Scripts\python.exe" goto :venv_existente

python --version >nul 2>&1
if not errorlevel 1 goto :achou_python

py -3 --version >nul 2>&1
if not errorlevel 1 goto :achou_py

goto :sem_python

:achou_python
set "PYCMD=python"
goto :criar_venv

:achou_py
set "PYCMD=py -3"
goto :criar_venv

REM ------------------------------------------------------------ 2. venv
:criar_venv
for /f "tokens=*" %%v in ('%PYCMD% --version 2^>^&1') do echo        %%v
echo.
echo  [2/3] Criando o ambiente do projeto...
%PYCMD% -m venv venv
if errorlevel 1 goto :falha_venv

:venv_existente
echo        Python do projeto encontrado.
echo.
echo  [2/3] O ambiente do projeto ja existe - reaproveitando.

:venv_pronto
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" goto :falha_venv
echo        Ambiente pronto  (pasta venv)

REM ------------------------------------------------------- 3. bibliotecas
echo.
echo  [3/3] Instalando as bibliotecas do projeto...
echo        (a lista completa esta em requirements.txt)
echo.
"%PY%" -m pip install --upgrade pip --disable-pip-version-check --quiet
"%PY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto :falha_pip

echo.
echo        Conferindo a instalacao...
"%PY%" -c "import fastapi, uvicorn, xlrd, matplotlib, sklearn" 2>nul
if errorlevel 1 goto :falha_pip
echo        Tudo certo.

REM ------------------------------------------- 4. interface ja compilada
echo.
echo  [extra] Conferindo a interface...
if exist "web_app\frontend\dist\index.html" goto :interface_ok

echo        A interface ainda nao esta compilada.
npm --version >nul 2>&1
if errorlevel 1 goto :sem_node
echo        Node.js encontrado - compilando agora (demora um pouco)...
call npm --prefix web_app/frontend install
call npm --prefix web_app/frontend run build
if errorlevel 1 goto :sem_node
echo        Interface compilada.
goto :fim_ok

:interface_ok
echo        Interface ja compilada - o Node.js nao e necessario.
goto :fim_ok

REM ================================================================ FIM OK
:fim_ok
echo.
echo  ===================================================================
echo    PRONTO! As dependencias estao instaladas.
echo.
echo    Para usar o projeto, de um duplo clique em:
echo        Iniciar Projeto.bat
echo  ===================================================================
echo.
choice /C SN /N /M "  Quer abrir o projeto agora?  [S = sim / N = nao]: "
if errorlevel 2 goto :sair
echo.
call "Iniciar Projeto.bat"
goto :sair

REM =============================================================== ERROS
:sem_python
echo.
echo  ===================================================================
echo    O Python nao foi encontrado neste computador.
echo  ===================================================================
echo.
winget --version >nul 2>&1
if errorlevel 1 goto :instalar_manual

echo    Posso instalar o Python automaticamente pela Loja da Microsoft.
echo.
choice /C SN /N /M "  Instalar o Python agora?  [S = sim / N = nao]: "
if errorlevel 2 goto :instalar_manual

echo.
echo    Instalando o Python... (aceite as janelas que aparecerem)
winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
echo.
echo  ===================================================================
echo    Python instalado.
echo.
echo    Agora FECHE esta janela e de um duplo clique no
echo    dependencias.bat outra vez - o Windows precisa reiniciar a
echo    janela para enxergar o Python recem-instalado.
echo  ===================================================================
echo.
pause
exit /b 1

:instalar_manual
echo.
echo    Instale o Python assim:
echo.
echo      1. abra  https://www.python.org/downloads/
echo      2. clique no botao amarelo "Download Python"
echo      3. execute o arquivo baixado
echo      4. IMPORTANTE: marque a caixinha
echo         "Add python.exe to PATH"  antes de clicar em Install
echo      5. termine a instalacao, feche esta janela e de um duplo
echo         clique no dependencias.bat outra vez
echo.
pause
exit /b 1

:falha_venv
echo.
echo    Nao consegui criar a pasta "venv".
echo    Tente executar este arquivo como administrador, ou copie o
echo    projeto para uma pasta simples, como  C:\TEIA
echo.
pause
exit /b 1

:falha_pip
echo.
echo    A instalacao das bibliotecas falhou.
echo.
echo    Causas mais comuns:
echo      - sem internet ou atras de um proxy/firewall
echo      - antivirus bloqueando o download
echo.
echo    Leia a mensagem em vermelho acima e tente de novo.
echo.
pause
exit /b 1

:sem_node
echo.
echo    A interface precisa ser compilada e o Node.js nao esta instalado.
echo.
echo    Duas saidas:
echo      - instale o Node.js em  https://nodejs.org  (versao 20 ou maior)
echo        e rode este arquivo de novo; ou
echo      - peca a pasta  web_app\frontend\dist  ja compilada para o
echo        grupo e copie para dentro do projeto.
echo.
echo    Enquanto isso, a interface do computador (Tkinter) funciona so
echo    com o Python:   venv\Scripts\python.exe iris_classifier\run_gui.py
echo.
pause
exit /b 1

:sair
exit /b 0
