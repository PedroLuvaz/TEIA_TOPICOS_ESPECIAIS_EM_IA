"""
Inicializador do projeto — sobe tudo e abre o navegador.

Pensado para a apresentacao: um duplo clique no `Iniciar Projeto.bat` (ou
`python iniciar.py`) e a interface abre pronta, sem precisar de dois
terminais nem lembrar de comandos.

Como funciona
-------------
Por padrao roda em **modo producao**: usa o build ja compilado do frontend
(`web_app/frontend/dist/`) e sobe apenas o servidor Python, que serve a API
e a interface na mesma porta. E o caminho mais rapido e o que menos tem o
que dar errado na hora de apresentar.

Se o build nao existir, o script compila sozinho (precisa do Node instalado)
e segue. Para desenvolver com hot reload, use `--dev`, que sobe o Vite em
paralelo.

Uso
---
    python iniciar.py               # modo producao (recomendado)
    python iniciar.py --dev         # com hot reload do Vite
    python iniciar.py --rebuild     # forca recompilar o frontend
    python iniciar.py --porta 9000  # troca a porta
    python iniciar.py --sem-browser # nao abre o navegador
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

RAIZ = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(RAIZ, 'web_app', 'frontend')
DIST = os.path.join(FRONTEND, 'dist')
INDEX = os.path.join(DIST, 'index.html')

# O console padrao do Windows usa cp1252 e engasga com acentos e simbolos.
# Reconfigurar para UTF-8 evita o UnicodeEncodeError no duplo clique.
for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass


def _console_unicode():
    """Testa se o console realmente consegue imprimir os simbolos bonitos."""
    try:
        '─✓•✗'.encode(sys.stdout.encoding or 'ascii')
        return True
    except (UnicodeEncodeError, LookupError):
        return False


UNICODE_OK = _console_unicode()

VERDE, AMARELO, VERMELHO, AZUL, CINZA, FIM = (
    '\033[92m', '\033[93m', '\033[91m', '\033[96m', '\033[90m', '\033[0m')

# Habilita as sequencias ANSI no console classico do Windows
if os.name == 'nt':
    try:
        import ctypes
        _h = ctypes.windll.kernel32.GetStdHandle(-11)
        _modo = ctypes.c_ulong()
        ctypes.windll.kernel32.GetConsoleMode(_h, ctypes.byref(_modo))
        ctypes.windll.kernel32.SetConsoleMode(_h, _modo.value | 0x0004)
        CORES_OK = True
    except Exception:
        CORES_OK = False
else:
    CORES_OK = sys.stdout.isatty()

# Simbolos com alternativa ASCII, para consoles antigos
S_OK = '✓' if UNICODE_OK else '[ok]'
S_INFO = '•' if UNICODE_OK else '[..]'
S_AVISO = '!' if UNICODE_OK else '[!]'
S_ERRO = '✗' if UNICODE_OK else '[x]'
S_PLAY = '▶' if UNICODE_OK else '>>'


def _cor(texto, cor):
    return f'{cor}{texto}{FIM}' if CORES_OK else texto


def ok(msg):
    print(f'  {_cor(S_OK, VERDE)} {msg}')


def info(msg):
    print(f'  {_cor(S_INFO, AZUL)} {msg}')


def aviso(msg):
    print(f'  {_cor(S_AVISO, AMARELO)} {msg}')


def erro(msg):
    print(f'  {_cor(S_ERRO, VERMELHO)} {msg}')


def cabecalho():
    titulo = 'TEIA - Reconhecimento de Padroes'
    sub = 'UEPB 2026 - Laboratorios 1 a 5 + Seminario'
    if UNICODE_OK:
        titulo = 'TEIA · Reconhecimento de Padrões'
        sub = 'UEPB 2026 · Laboratórios 1 a 5 + Seminário'
        h, v, tl, tr, bl, br = '─', '│', '┌', '┐', '└', '┘'
    else:
        h, v, tl, tr, bl, br = '-', '|', '+', '+', '+', '+'

    largura = 51
    print()
    print(_cor('  ' + tl + h * largura + tr, CINZA))
    print(_cor('  ' + v, CINZA) + '  ' + titulo.ljust(largura - 2)
          + _cor(v, CINZA))
    print(_cor('  ' + v, CINZA) + _cor('  ' + sub.ljust(largura - 2), CINZA)
          + _cor(v, CINZA))
    print(_cor('  ' + bl + h * largura + br, CINZA))
    print()


def python_do_venv():
    """Caminho do Python do venv, se existir; senao o interpretador atual."""
    candidatos = [
        os.path.join(RAIZ, 'venv', 'Scripts', 'python.exe'),   # Windows
        os.path.join(RAIZ, 'venv', 'bin', 'python'),           # Unix
        os.path.join(RAIZ, '.venv', 'Scripts', 'python.exe'),
        os.path.join(RAIZ, '.venv', 'bin', 'python'),
    ]
    for c in candidatos:
        if os.path.isfile(c):
            return c
    return sys.executable


def porta_livre(porta):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', porta)) != 0


def achar_porta(inicial):
    """Devolve a primeira porta livre a partir da informada."""
    for p in range(inicial, inicial + 20):
        if porta_livre(p):
            return p
    return None


def comando_npm():
    """No Windows o npm e um .cmd — shutil.which resolve isso."""
    return shutil.which('npm') or shutil.which('npm.cmd')


def checar_dependencias(py):
    """Confere se o FastAPI esta disponivel antes de tentar subir."""
    try:
        subprocess.run(
            [py, '-c', 'import fastapi, uvicorn'],
            check=True, capture_output=True, cwd=RAIZ)
        return True
    except subprocess.CalledProcessError:
        erro('Dependências Python ausentes (fastapi / uvicorn).')
        print()
        print(f'      Instale com:  {_cor(f"{py} -m pip install -r requirements.txt", AZUL)}')
        print()
        return False


def compilar_frontend():
    """Roda `npm run build`, instalando as dependencias se necessario."""
    npm = comando_npm()
    if not npm:
        erro('Node.js/npm não encontrado — necessário para compilar a interface.')
        print()
        print('      Baixe em: https://nodejs.org  (versão 20 ou superior)')
        print(f'      Ou rode em outra máquina e copie a pasta '
              f'{_cor("web_app/frontend/dist", AZUL)}.')
        print()
        return False

    if not os.path.isdir(os.path.join(FRONTEND, 'node_modules')):
        info('Instalando dependências da interface (só na primeira vez)…')
        r = subprocess.run([npm, 'install'], cwd=FRONTEND)
        if r.returncode != 0:
            erro('Falha ao instalar as dependências da interface.')
            return False
        ok('Dependências instaladas.')

    info('Compilando a interface…')
    r = subprocess.run([npm, 'run', 'build'], cwd=FRONTEND)
    if r.returncode != 0:
        erro('Falha ao compilar a interface.')
        return False
    ok('Interface compilada.')
    return True


def esperar_servidor(url, timeout=90):
    """Aguarda o /api/health responder — evita abrir o browser cedo demais."""
    limite = time.time() + timeout
    while time.time() < limite:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.4)
    return False


def main():
    p = argparse.ArgumentParser(
        description='Sobe o projeto e abre o navegador.')
    p.add_argument('--dev', action='store_true',
                   help='modo desenvolvimento, com hot reload do Vite')
    p.add_argument('--rebuild', action='store_true',
                   help='recompila a interface antes de subir')
    p.add_argument('--porta', type=int, default=8000,
                   help='porta do servidor (padrão: 8000)')
    p.add_argument('--sem-browser', action='store_true',
                   help='não abre o navegador automaticamente')
    args = p.parse_args()

    cabecalho()

    py = python_do_venv()
    if py != sys.executable:
        ok(f'Ambiente virtual encontrado.')
    if not checar_dependencias(py):
        return 1

    # --- Interface compilada -------------------------------------------
    if args.dev:
        info('Modo desenvolvimento — a interface virá do Vite.')
    else:
        if args.rebuild or not os.path.isfile(INDEX):
            if not os.path.isfile(INDEX):
                aviso('Interface ainda não compilada.')
            if not compilar_frontend():
                return 1
        else:
            ok('Interface compilada encontrada.')

    # --- Porta ----------------------------------------------------------
    porta = achar_porta(args.porta)
    if porta is None:
        erro(f'Nenhuma porta livre a partir da {args.porta}.')
        return 1
    if porta != args.porta:
        aviso(f'Porta {args.porta} ocupada — usando {porta}.')

    processos = []
    url_app = f'http://127.0.0.1:{porta}'

    try:
        # --- Backend ----------------------------------------------------
        cmd = [py, '-m', 'uvicorn', 'web_app.backend.main:app',
               '--host', '127.0.0.1', '--port', str(porta)]
        if args.dev:
            cmd.append('--reload')
        info(f'Subindo o servidor na porta {porta}…')
        processos.append(subprocess.Popen(cmd, cwd=RAIZ))

        # --- Vite (só no modo dev) --------------------------------------
        if args.dev:
            npm = comando_npm()
            if not npm:
                erro('npm não encontrado — o modo --dev precisa do Node.')
                return 1
            porta_vite = achar_porta(5173) or 5173
            info(f'Subindo o Vite na porta {porta_vite}…')
            processos.append(subprocess.Popen(
                [npm, 'run', 'dev', '--', '--port', str(porta_vite)],
                cwd=FRONTEND))
            url_app = f'http://localhost:{porta_vite}'

        # --- Espera e abre o navegador ----------------------------------
        if not esperar_servidor(f'http://127.0.0.1:{porta}/api/health'):
            erro('O servidor não respondeu a tempo.')
            return 1
        ok('Servidor no ar.')

        if args.dev:
            time.sleep(2.5)  # dá um respiro para o Vite terminar de subir

        print()
        print(f'     {_cor(S_PLAY, VERDE)}  {_cor(url_app, AZUL)}')
        print(f'     {_cor("API/docs:", CINZA)} '
              f'{_cor(f"http://127.0.0.1:{porta}/docs", CINZA)}')
        print()
        print(_cor('     Pressione Ctrl+C nesta janela para encerrar.', CINZA))
        print()

        if not args.sem_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url_app)).start()

        # Encerra se qualquer processo morrer
        while True:
            for proc in processos:
                if proc.poll() is not None:
                    aviso('Um dos serviços encerrou.')
                    return proc.returncode or 0
            time.sleep(0.5)

    except KeyboardInterrupt:
        print()
        info('Encerrando…')
        return 0
    finally:
        for proc in processos:
            if proc.poll() is None:
                proc.terminate()
        for proc in processos:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        ok('Tudo encerrado.')
        print()


if __name__ == '__main__':
    sys.exit(main())
