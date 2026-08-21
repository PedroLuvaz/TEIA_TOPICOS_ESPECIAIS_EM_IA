"""
API web do projeto TEIA — Reconhecimento de Padroes.

Expõe, via JSON, os mesmos experimentos da interface desktop (Tkinter),
reutilizando integralmente os modelos em Python puro de `iris_classifier/`.
Nenhuma matematica e reimplementada aqui.

Execucao (a partir da raiz do projeto):
    uvicorn web_app.backend.main:app --reload --port 8000
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core import BASE_DIR
from .routers import (bayes, classificar, dataset, distancia_minima,
                      floresta, lab5, metricas, perceptron_delta)

app = FastAPI(
    title='TEIA · Reconhecimento de Padroes',
    description='API dos laboratorios de Topicos Especiais em IA (UEPB)',
    version='2.0.0',
)

# Em desenvolvimento o Vite serve o frontend em outra porta.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

for r in (dataset, classificar, distancia_minima, perceptron_delta,
          metricas, bayes, lab5, floresta):
    app.include_router(r.router)


@app.get('/api/health')
def health():
    """Checagem rapida de disponibilidade e de dependencias opcionais."""
    try:
        import sklearn  # noqa: F401
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False
    return {'status': 'ok', 'sklearn': sklearn_ok}


# --- Producao: servir o build do frontend (se existir) ---------------------
DIST = os.path.join(BASE_DIR, 'web_app', 'frontend', 'dist')

if os.path.isdir(DIST):
    app.mount('/assets', StaticFiles(directory=os.path.join(DIST, 'assets')),
              name='assets')

    @app.get('/{caminho:path}')
    def spa(caminho: str):
        """Entrega o index.html para qualquer rota — roteamento no cliente."""
        arquivo = os.path.join(DIST, caminho)
        if caminho and os.path.isfile(arquivo):
            return FileResponse(arquivo)
        return FileResponse(os.path.join(DIST, 'index.html'))
