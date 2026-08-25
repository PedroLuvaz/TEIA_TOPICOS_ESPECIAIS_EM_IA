"""
Gera o GUIA_DO_PROFESSOR.pdf, na raiz do projeto.

O PDF e o unico documento que o professor precisa abrir: ensina a instalar e
executar o projeto em dois cliques e mostra onde encontrar toda a documentacao,
laboratorio por laboratorio.

Por que um script e nao um Markdown convertido
----------------------------------------------
O conteudo mora aqui, em vez de sair de um `.md` por conversao, para que o
projeto nao dependa de pandoc/wkhtmltopdf — nenhum dos dois esta instalado numa
maquina comum. O unico requisito e o `reportlab`, que e Python puro:

    pip install reportlab

Uso:
    python ferramentas/gerar_guia_pdf.py
"""
import os
import sys

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)
except ImportError:
    print('Este script precisa do reportlab. Instale com:')
    print('    pip install reportlab')
    sys.exit(1)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(RAIZ, 'GUIA_DO_PROFESSOR.pdf')

# Paleta do projeto: ardosia com destaque ambar.
TINTA = colors.HexColor('#0f172a')
TEXTO = colors.HexColor('#334155')
SUAVE = colors.HexColor('#64748b')
DESTAQUE = colors.HexColor('#b45309')
LINHA = colors.HexColor('#e2e8f0')
FUNDO = colors.HexColor('#f8fafc')
FUNDO_DESTAQUE = colors.HexColor('#fef3c7')

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
_base = getSampleStyleSheet()

TITULO = ParagraphStyle(
    'Titulo', parent=_base['Title'], fontName='Helvetica-Bold',
    fontSize=23, leading=27, textColor=TINTA, spaceAfter=2, alignment=TA_CENTER)

SUBTITULO = ParagraphStyle(
    'Subtitulo', parent=_base['Normal'], fontName='Helvetica',
    fontSize=10.5, leading=15, textColor=SUAVE, alignment=TA_CENTER,
    spaceAfter=14)

SECAO = ParagraphStyle(
    'Secao', parent=_base['Heading1'], fontName='Helvetica-Bold',
    fontSize=14, leading=18, textColor=TINTA, spaceBefore=12, spaceAfter=7)

SUBSECAO = ParagraphStyle(
    'Subsecao', parent=_base['Heading2'], fontName='Helvetica-Bold',
    fontSize=11, leading=15, textColor=DESTAQUE, spaceBefore=11, spaceAfter=4)

CORPO = ParagraphStyle(
    'Corpo', parent=_base['Normal'], fontName='Helvetica',
    fontSize=10, leading=14.5, textColor=TEXTO, spaceAfter=6)

PASSO = ParagraphStyle(
    'Passo', parent=CORPO, leftIndent=16, bulletIndent=3, spaceAfter=5)

CELULA = ParagraphStyle(
    'Celula', parent=CORPO, fontSize=9, leading=13, spaceAfter=0)

CELULA_FORTE = ParagraphStyle(
    'CelulaForte', parent=CELULA, fontName='Helvetica-Bold', textColor=TINTA)

CODIGO = ParagraphStyle(
    'Codigo', parent=CORPO, fontName='Courier-Bold', fontSize=10.5,
    leading=14, textColor=TINTA, spaceAfter=0)

NOTA = ParagraphStyle(
    'Nota', parent=CORPO, fontSize=9.5, leading=13.5, spaceAfter=0)


# ---------------------------------------------------------------------------
# Blocos reutilizaveis
# ---------------------------------------------------------------------------
def p(texto, estilo=CORPO):
    return Paragraph(texto, estilo)


def passos(itens):
    """Lista numerada."""
    return [Paragraph(f'{i}. {t}', PASSO) for i, t in enumerate(itens, 1)]


def caixa(conteudo, fundo=FUNDO, borda=LINHA):
    """Um bloco destacado, com fundo e borda finos."""
    t = Table([[conteudo]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), fundo),
        ('BOX', (0, 0), (-1, -1), 0.6, borda),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def tabela(cabecalho, linhas, larguras):
    """Tabela com cabecalho em ardosia e zebrado discreto."""
    dados = [[p(c, CELULA_FORTE) for c in cabecalho]]
    for linha in linhas:
        dados.append([p(c, CELULA) for c in linha])

    t = Table(dados, colWidths=larguras, repeatRows=1)
    estilo = [
        ('BACKGROUND', (0, 0), (-1, 0), FUNDO),
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, SUAVE),
        ('LINEBELOW', (0, 1), (-1, -2), 0.4, LINHA),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(estilo))
    return t


def rodape(canvas, doc):
    """Numero da pagina e identificacao, em todas as folhas."""
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(SUAVE)
    canvas.drawString(
        22 * mm, 12 * mm,
        'Reconhecimento de Padrões  ·  Tópicos Especiais em IA  ·  UEPB')
    canvas.drawRightString(188 * mm, 12 * mm, f'{doc.page}')
    canvas.setStrokeColor(LINHA)
    canvas.setLineWidth(0.4)
    canvas.line(22 * mm, 16 * mm, 188 * mm, 16 * mm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Conteudo
# ---------------------------------------------------------------------------
def construir():
    e = []   # elementos do documento

    # ---------------------------------------------------------- cabecalho
    e.append(p('Guia de Execução do Projeto', TITULO))
    e.append(p('Reconhecimento de Padrões &mdash; Tópicos Especiais em '
               'Inteligência Artificial<br/>'
               'Universidade Estadual da Paraíba &nbsp;·&nbsp; 2026<br/>'
               'Erick Nathan &nbsp;·&nbsp; Laura Barbosa &nbsp;·&nbsp; '
               'Pedro Lucas', SUBTITULO))

    e.append(caixa([
        p('<b>O projeto abre em dois cliques.</b>', NOTA),
        Spacer(1, 5),
        p('1. &nbsp;duplo clique em &nbsp;<font face="Courier-Bold">'
          'dependencias.bat</font>&nbsp; &mdash; instala tudo, só na primeira '
          'vez;', NOTA),
        p('2. &nbsp;duplo clique em &nbsp;<font face="Courier-Bold">'
          'Iniciar Projeto.bat</font>&nbsp; &mdash; o navegador abre sozinho '
          'com a aplicação.', NOTA),
        Spacer(1, 5),
        p('Só o <b>Python</b> precisa estar instalado &mdash; e o próprio '
          'instalador se oferece para instalá-lo, caso falte.', NOTA),
    ], fundo=FUNDO_DESTAQUE, borda=DESTAQUE))

    # ------------------------------------------------------ 1. requisitos
    e.append(p('1. &nbsp;O que precisa estar instalado', SECAO))
    e.append(tabela(
        ['Programa', 'Precisa?', 'Observação'],
        [['Python 3.10 ou mais novo', 'Sim',
          'Se não estiver instalado, o <font face="Courier">dependencias.bat'
          '</font> se oferece para instalar'],
         ['Node.js', 'Não',
          'A interface já vem compilada dentro do projeto'],
         ['R', 'Não',
          'Só para o teste extra de normalidade multivariada; sem ele o '
          'aplicativo usa os valores já calculados']],
        [52 * mm, 20 * mm, 93 * mm]))
    e.append(Spacer(1, 6))
    e.append(p('Também é preciso <b>conexão com a internet</b> na primeira '
               'execução, porque o instalador baixa as bibliotecas.'))

    # -------------------------------------------------------- 2. instalar
    e.append(p('2. &nbsp;Passo 1 &mdash; Instalar (só uma vez)', SECAO))
    e.append(caixa(p(
        '<b>Recebeu o projeto num arquivo .zip?</b> Antes de tudo, clique com '
        'o botão direito nele e escolha <b>Extrair tudo</b>. Os programas '
        'precisam ser executados a partir da pasta extraída &mdash; abertos de '
        'dentro do .zip, eles não encontram os arquivos do projeto.', NOTA)))
    e.append(Spacer(1, 8))
    e.extend(passos([
        'Abra a pasta do projeto.',
        'Dê um <b>duplo clique</b> no arquivo '
        '<font face="Courier-Bold">dependencias.bat</font>.',
        'Vai abrir uma janela preta. Aperte qualquer tecla quando ela pedir.',
        'Espere. Na primeira vez leva de 2 a 5 minutos &mdash; ele está '
        'baixando as bibliotecas.',
        'Ao terminar, aparece a mensagem <b>&ldquo;PRONTO! As dependencias '
        'estao instaladas&rdquo;</b> e a pergunta se deseja abrir o projeto '
        'agora. Digite <b>S</b> para abrir na hora ou <b>N</b> para abrir '
        'depois.',
    ]))

    e.append(p('Se aparecer &ldquo;O Python nao foi encontrado&rdquo;',
               SUBSECAO))
    e.append(p('O próprio instalador resolve, de duas formas:'))
    e.append(p(
        '<b>Automática:</b> ele pergunta &ldquo;Instalar o Python '
        'agora?&rdquo;. Digite <b>S</b>, aceite as janelas que aparecerem, '
        '<b>feche a janela preta</b> e dê um duplo clique no '
        '<font face="Courier">dependencias.bat</font> outra vez.', PASSO))
    e.append(p(
        '<b>Manual:</b> baixe em <font face="Courier">'
        'python.org/downloads</font> e execute o arquivo. '
        '<b>Marque a caixinha &ldquo;Add python.exe to PATH&rdquo;</b> antes '
        'de clicar em <i>Install</i> &mdash; sem ela o Windows não encontra o '
        'Python depois. Terminada a instalação, rode o '
        '<font face="Courier">dependencias.bat</font> de novo.', PASSO))

    e.append(Spacer(1, 4))
    e.append(caixa(p(
        '<b>Aviso azul do Windows?</b> Se aparecer &ldquo;O Windows protegeu '
        'o computador&rdquo;, clique em <b>Mais informações</b> e depois em '
        '<b>Executar assim mesmo</b>. Os arquivos <font face="Courier">.bat'
        '</font> são texto puro: dá para abri-los no Bloco de Notas e ler '
        'tudo o que fazem.', NOTA)))

    # ----------------------------------------------------------- 3. abrir
    e.append(p('3. &nbsp;Passo 2 &mdash; Abrir o projeto', SECAO))
    e.extend(passos([
        'Duplo clique em <font face="Courier-Bold">Iniciar Projeto.bat</font>.',
        'A janela preta mostra o progresso e, em poucos segundos, o '
        '<b>navegador abre sozinho</b> em '
        '<font face="Courier">http://127.0.0.1:8000</font>.',
        'Pronto: a aplicação está rodando.',
    ]))
    e.append(Spacer(1, 4))
    e.append(caixa(p(
        '<b>Não feche a janela preta</b> enquanto estiver usando o projeto '
        '&mdash; é ela que mantém o programa no ar. Para encerrar, feche o '
        'navegador e aperte <b>Ctrl + C</b> na janela preta.', NOTA)))

    # -------------------------------------------------- 4. dentro do app
    e.append(p('4. &nbsp;O que ver dentro do aplicativo', SECAO))
    e.append(p('A barra da esquerda tem as telas. Cada uma corresponde a uma '
               'parte da entrega:'))

    e.append(p('Classificar &mdash; a tela principal', SUBSECAO))
    e.append(p(
        'É onde o modelo é <b>escolhido</b> e <b>parametrizado</b>. Escolha a '
        'base de dados, escolha entre os sete classificadores e ajuste os '
        'hiperparâmetros &mdash; os controles mudam conforme o modelo. Abaixo '
        'aparecem as métricas, a matriz de confusão e as regiões de decisão; '
        'clicando no gráfico, o ponto é classificado na hora.'))
    e.append(caixa(p(
        '<b>Sugestão:</b> escolha <b>Sépalas</b> em <i>Atributos</i>. Nas '
        'pétalas quase todo modelo acerta 100% e as diferenças somem; nas '
        'sépalas as classes se sobrepõem e dá para ver cada modelo se '
        'comportando de um jeito.', NOTA)))
    e.append(Spacer(1, 8))

    e.append(p('Importar .txt &mdash; a base de dados do usuário', SUBSECAO))
    e.append(p(
        'No painel <i>Configuração do experimento</i>, o botão <b>Importar '
        '.txt</b> carrega qualquer arquivo de texto com uma amostra por linha. '
        'A tela já abre com um exemplo carregado, mostrando o formato '
        'esperado, e exibe como o arquivo foi entendido &mdash; separador, '
        'cabeçalho e coluna da classe &mdash; tudo corrigível antes de '
        'confirmar. Depois de importada, a base vale em todas as telas.'))
    e.append(Spacer(1, 8))

    e.append(p('Métricas Avançadas &mdash; qualidade e significância',
               SUBSECAO))
    e.append(p(
        'Quatro abas: validação cruzada (média, desvio e intervalo de '
        'confiança), split único, testes de significância (McNemar, bootstrap '
        'pareado e permutação, além do teste Z de Kappa) e a matriz de '
        'confusão editável. É aqui que os sete modelos são comparados entre '
        'si, par a par.'))
    e.append(Spacer(1, 8))

    e.append(p('Florestas Aleatórias &mdash; o modelo do seminário', SUBSECAO))
    e.append(p(
        'O modelo apresentado no seminário, com as árvores navegáveis uma a '
        'uma, o erro <i>out-of-bag</i> e a importância de cada atributo. Ele '
        'também aparece no catálogo da tela <i>Classificar</i>.'))

    # ------------------------------------------------------- 5. problemas
    e.append(p('5. &nbsp;Se alguma coisa der errado', SECAO))
    e.append(tabela(
        ['O que aconteceu', 'O que fazer'],
        [['A janela preta abre e fecha na hora',
          'Rode o <font face="Courier">dependencias.bat</font> primeiro; se '
          'já rodou, clique com o botão direito no arquivo e escolha '
          '<i>Executar como administrador</i>'],
         ['&ldquo;Dependências Python ausentes&rdquo;',
          'Rode o <font face="Courier">dependencias.bat</font>'],
         ['O navegador não abriu sozinho',
          'Abra o navegador e digite '
          '<font face="Courier">http://127.0.0.1:8000</font>'],
         ['&ldquo;Porta 8000 ocupada&rdquo;',
          'Não é problema: o programa procura a próxima porta livre e mostra '
          'o endereço certo na janela preta'],
         ['A página abre em branco',
          'Espere 5 segundos e atualize com <b>F5</b>'],
         ['Nada funciona',
          'Apague a pasta <font face="Courier">venv</font> e rode o '
          '<font face="Courier">dependencias.bat</font> de novo']],
        [58 * mm, 107 * mm]))

    # --------------------------------------------------- 6. documentacao
    e.append(p('6. &nbsp;Onde encontrar a documentação', SECAO))
    e.append(p(
        'Toda a documentação escrita está na pasta <font face="Courier">docs/'
        '</font> do projeto, em arquivos de texto que abrem em qualquer editor '
        '&mdash; e formatados quando lidos no GitHub. O índice completo, com '
        'a descrição de cada documento, está em '
        '<font face="Courier">docs/README.md</font>.'))

    e.append(p('Comece por aqui', SUBSECAO))
    e.append(tabela(
        ['Documento', 'Para quê'],
        [['docs/defesa_projeto.md',
          '<b>O guia único do projeto.</b> Requisitos da entrega e onde cada '
          'um foi atendido, teoria dos sete modelos, métricas, testes de '
          'significância, arquitetura e resultados medidos'],
         ['docs/README.md',
          'Índice de toda a documentação, organizado por laboratório'],
         ['TUTORIAL_RODAR_PROJETO.md',
          'A versão em texto deste guia, na raiz do projeto'],
         ['README.md',
          'Visão geral do repositório e estrutura de pastas']],
        [62 * mm, 103 * mm]))

    e.append(p('Por laboratório', SUBSECAO))
    e.append(tabela(
        ['Laboratório', 'Documentos'],
        [['Lab 1 &mdash; Distância Mínima',
          'docs/teoria_completa.md (§1 a §10)<br/>docs/formulario.md'],
         ['Lab 2 &mdash; Perceptron e Regra Delta',
          'docs/teoria_completa.md (§11 a §16)<br/>docs/formulario.md'],
         ['Lab 3 &mdash; Métricas e significância',
          'docs/lab_03/teoria_lab03.md<br/>'
          'docs/lab_03/testes_significancia.md<br/>'
          'docs/lab_03/item_02.md &nbsp;·&nbsp; item_03.md'],
         ['Lab 4 &mdash; Bayes e Naive Bayes',
          'docs/lab_04/teoria_lab04.md<br/>'
          'docs/lab_04/relatorio_experimentos.md'],
         ['Lab 5 &mdash; MLP e backpropagation',
          'docs/lab_05/teoria_lab05.md<br/>'
          'docs/lab_05/relatorio_experimentos.md'],
         ['Seminário &mdash; Florestas Aleatórias',
          'docs/seminario_florestas_aleatorias.md<br/>'
          'docs/seminario_dataset_fim_de_semana.md'],
         ['Entrega final',
          'docs/defesa_projeto.md<br/>docs/classificar_modelos.md<br/>'
          'docs/importar_dados_txt.md<br/>docs/interface_web.md']],
        [58 * mm, 107 * mm]))

    e.append(Spacer(1, 10))
    e.append(p('Onde está o código de cada assunto', SUBSECAO))
    e.append(tabela(
        ['Assunto', 'Arquivo'],
        [['Álgebra linear em Python puro',
          'iris_classifier/core/math_utils.py'],
         ['Distância Mínima', 'iris_classifier/models/classifier.py'],
         ['Perceptron (binário e Um-Contra-Todos)',
          'iris_classifier/models/perceptron.py'],
         ['Regra Delta', 'iris_classifier/models/delta_rule.py'],
         ['Bayes Ótimo e Naive Bayes',
          'iris_classifier/models/bayes_classifier.py'],
         ['Rede feedforward e backpropagation',
          'iris_classifier/models/mlp_backprop.py<br/>'
          'iris_classifier/models/mlp_multiclasse.py'],
         ['Florestas Aleatórias (seminário)',
          'iris_classifier/models/random_forest.py<br/>'
          'iris_classifier/models/floresta_categorica.py'],
         ['Métricas e testes de significância',
          'iris_classifier/evaluation/metricas_avancadas.py<br/>'
          'iris_classifier/evaluation/testes_significancia.py'],
         ['Leitura da base .txt do usuário',
          'iris_classifier/data/leitor_texto.py'],
         ['Catálogo de modelos e parâmetros',
          'web_app/backend/modelos.py']],
        [58 * mm, 107 * mm]))

    # ------------------------------------------------------------- outros
    e.append(p('7. &nbsp;Outras formas de executar', SECAO))
    e.append(p('Opcionais &mdash; a aplicação web já cobre tudo o que a '
               'entrega pede.'))
    e.append(p('Interface para o computador, sem navegador:'))
    e.append(caixa(p(
        'venv\\Scripts\\python.exe iris_classifier\\run_gui.py', CODIGO)))
    e.append(Spacer(1, 6))
    e.append(p('Todos os experimentos no terminal, salvando os gráficos em '
               '<font face="Courier">outputs/</font>:'))
    e.append(caixa(p(
        'venv\\Scripts\\python.exe iris_classifier\\main.py', CODIGO)))

    e.append(Spacer(1, 8))
    e.append(caixa(p(
        'Projeto disponível em <font face="Courier">'
        'github.com/PedroLuvaz/TEIA_TOPICOS_ESPECIAIS_EM_IA</font><br/>'
        'Todos os classificadores foram implementados do zero, em Python '
        'puro, sem bibliotecas de aprendizado de máquina.', NOTA)))
    return e


def main():
    doc = SimpleDocTemplate(
        SAIDA, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=22 * mm,
        title='Guia de Execução do Projeto — Reconhecimento de Padrões',
        author='Erick Nathan, Laura Barbosa e Pedro Lucas',
        subject='Como instalar, executar e onde encontrar a documentação')

    doc.build(construir(), onFirstPage=rodape, onLaterPages=rodape)
    print(f'PDF gerado: {SAIDA}')
    print(f'{os.path.getsize(SAIDA) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
