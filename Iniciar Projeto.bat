@echo off
REM ===================================================================
REM  TEIA - Reconhecimento de Padroes
REM  Duplo clique aqui para subir o projeto e abrir o navegador.
REM ===================================================================
title TEIA - Reconhecimento de Padroes
cd /d "%~dp0"

REM Console em UTF-8 para os acentos sairem certos (>nul esconde o aviso).
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Usa o Python do venv, se existir; senao cai para o Python do sistema.
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" iniciar.py %*

REM Se algo falhou, mantem a janela aberta para a mensagem ser lida.
if errorlevel 1 (
    echo.
    echo  Algo deu errado. Leia a mensagem acima.
    echo.
    pause
)
