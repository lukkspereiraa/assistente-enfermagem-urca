import time
import os
import re
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types

# 1. Inicialização e Configurações
load_dotenv()
chave_gemini = os.getenv("GEMINI_API_KEY")
if not chave_gemini:
    raise ValueError("ERRO: Chave GEMINI_API_KEY não encontrada nas variáveis de ambiente!")

warnings.filterwarnings("ignore")

app = FastAPI(title="API Assistente Enfermagem URCA (Render Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Carregar todos os PDFs diretamente para a memória RAM (Leve: < 50MB RAM)
print("📚 A carregar PDFs da URCA na memória...")
paginas_memoria = []

for raiz, _, arquivos in os.walk("."):
    for arquivo in sorted(arquivos):
        if arquivo.lower().endswith(".pdf"):
            caminho_pdf = os.path.join(raiz, arquivo)
            try:
                leitor = PdfReader(caminho_pdf)
                for num_pagina, pagina in enumerate(leitor.pages, start=1):
                    texto_pagina = pagina.extract_text() or ""
                    if texto_pagina.strip():
                        texto_formatado = f"=== DOCUMENTO: {arquivo} (PÁGINA {num_pagina}) ===\n{texto_pagina}"
                        paginas_memoria.append({
                            "content": texto_formatado,
                            "content_lower": texto_pagina.lower(),
                            "source": arquivo,
                            "page": num_pagina
                        })
            except Exception as e:
                print(f"⚠️ Erro ao ler {arquivo}: {e}")

print(f"✅ {len(paginas_memoria)} páginas carregadas diretamente dos PDFs!")

# 3. Conectar à API do Gemini e listar modelos disponíveis
print("🔌 A conectar com a Google...")
client = genai.Client(api_key=chave_gemini)

modelos_conta = []
try:
    for m in client.models.list():
        nome = m.name.replace("models/", "")
        if "gemini" in nome and not any(x in nome for x in ["embedding", "imagen", "veo", "tts", "audio", "vision"]):
            modelos_conta.append(nome)
except Exception as e:
    print(f"⚠️ Aviso ao listar modelos: {e}")

modelos_prioritarios = ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash"]
modelos_para_tentar = [m for m in modelos_prioritarios if m in modelos_conta] + [m for m in modelos_conta if m not in modelos_prioritarios]
if not modelos_para_tentar:
    modelos_para_tentar = modelos_prioritarios

# 4. Modelo de Dados da Requisição
class PedidoChat(BaseModel):
    pergunta: str
    historico: list[str] = []

# 5. Rota Principal (Entrega o index.html)
@app.get("/")
def abrir_interface():
    return FileResponse("index.html")

# 6. Rota da API do Chat (POST /chat)
@app.post("/chat")
def conversar(pedido: PedidoChat):
    pergunta = pedido.pergunta.strip()
    if not pergunta:
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    pergunta_lower = pergunta.lower()
    paginas_selecionadas = []
    fontes_selecionadas = []

    palavras_grade = [
        "horário", "horario", "horarios", "semestre", "semstre", "semetrte", 
        "período", "periodo", "pereildo", "perildos", "matriz", "disciplina", 
        "diciplina", "matéria", "materia", "professor", "professora", "grade", 
        "ementa", "emenda", "cadeira", "optativa", "pré-requisito", "pre-requisito",
        "segunda", "terça", "terca", "quarta", "quinta", "sexta", "manha", "manhã", "tarde"
    ]
    
    palavras_complementares = [
        "complementar", "complementares", "certificado", "certificados", 
        "horas", "palestra", "congresso", "monitoria", "extensão", "extensao", "artigo"
    ]

    # Gatilho 1: Horários e Matriz Curricular
    if any(p in pergunta_lower for p in palavras_grade):
        for item in paginas_memoria:
            nome_arq = item["source"].lower()
            if "horário" in nome_arq or "horario" in nome_arq or "matriz" in nome_arq:
                paginas_selecionadas.append(item["content"])
                fontes_selecionadas.append(f"{item['source']} (Pág. {item['page']})")

    # Gatilho 2: Atividades Complementares
    if any(p in pergunta_lower for p in palavras_complementares):
        for item in paginas_memoria:
            nome_arq = item["source"].lower()
            if "complementar" in nome_arq or "atividade" in nome_arq:
                if item["content"] not in paginas_selecionadas:
                    paginas_selecionadas.append(item["content"])
                    fontes_selecionadas.append(f"{item['source']} (Pág. {item['page']})")

    # Busca Leve por Relevância de Palavras (substitui o ChromaDB para Manual Acadêmico e POP)
    termos_busca = [t for t in re.findall(r'\w+', pergunta_lower) if len(t) > 3]
    ranking_paginas = []
    for item in paginas_memoria:
        if item["content"] not in paginas_selecionadas:
            pontos = sum(1 for termo in termos_busca if termo in item["content_lower"])
            # Bônus para o POP dos calouros e Manual em dúvidas gerais
            if "pop" in item["source"].lower():
                pontos += 0.5
            ranking_paginas.append((pontos, item))

    ranking_paginas.sort(key=lambda x: x[0], reverse=True)

    # Adiciona as 8 páginas mais relevantes que ainda não entraram
    for _, item in ranking_paginas[:8]:
        paginas_selecionadas.append(item["content"])
        fontes_selecionadas.append(f"{item['source']} (Pág. {item['page']})")

    contexto = "\n\n".join(paginas_selecionadas)
    historico_texto = "\n".join(pedido.historico[-4:])

    prompt = f"""Você é o assistente virtual oficial da URCA (Universidade Regional do Cariri), focado no curso de Enfermagem (Campus Iguatu).
    
    Regras importantes:
    1. Responda de forma amigável, organizada e direta, baseando-se APENAS nos documentos oficiais abaixo.
    2. Atenção: nos documentos, '1º Período', 'I Período', '1º Semestre' ou 'S1' significam a mesma coisa. Mesmo que o aluno escreva com erros de digitação, entenda a intenção dele.
    3. Use o Histórico da Conversa para entender perguntas de continuação.

    Histórico recente da conversa:
    {historico_texto}

    Documentos oficiais da URCA:
    {contexto}
    
    Pergunta atual do aluno: {pergunta}"""

    for modelo_atual in modelos_para_tentar:
        try:
            resposta = client.models.generate_content(
                model=modelo_atual,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return {
                "resposta": resposta.text,
                "modelo": modelo_atual,
                "fontes": fontes_selecionadas[:6]
            }
        except Exception as e:
            erro_str = str(e)
            if "503" in erro_str or "UNAVAILABLE" in erro_str or "429" in erro_str:
                time.sleep(1)
                continue
            elif "404" in erro_str or "NOT_FOUND" in erro_str or "400" in erro_str:
                continue
            else:
                time.sleep(1)

    raise HTTPException(status_code=503, detail="Servidores da Google temporariamente ocupados.")