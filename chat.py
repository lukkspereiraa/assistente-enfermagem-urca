import time
import os
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types

# 1. Carregar variáveis
load_dotenv()
chave_gemini = os.getenv("GEMINI_API_KEY")
if not chave_gemini:
    raise ValueError("ERRO: Chave não encontrada no ficheiro .env!")

warnings.filterwarnings("ignore")

# 2. Conectar à API e ordenar modelos mais rápidos primeiro
print("🔌 A conectar com a Google...")
client = genai.Client(api_key=chave_gemini)

modelos_conta = []
try:
    for m in client.models.list():
        nome = m.name.replace("models/", "")
        if "gemini" in nome and not any(x in nome for x in ["embedding", "imagen", "veo", "tts", "audio", "vision"]):
            modelos_conta.append(nome)
except Exception as e:
    print(f"❌ Erro ao conectar na API: {e}")
    exit()

modelos_prioritarios = ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash"]
modelos_para_tentar = [m for m in modelos_prioritarios if m in modelos_conta] + [m for m in modelos_conta if m not in modelos_prioritarios]

# 3. Carregar ChromaDB
print("🧠 A carregar o cérebro do bot...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = Chroma(
    persist_directory="./chroma_db_urca",
    embedding_function=embeddings_model
)

# Pega todos os documentos da base para permitir filtragem exata por arquivo quando necessário
todos_os_dados = vector_store.get()
total_paginas = len(todos_os_dados["documents"])
print(f"📊 Raio-X do ChromaDB: {total_paginas} páginas inteiras carregadas.")

# Memória das últimas interações
historico_conversa = []

print("\n🤖 Assistente da URCA pronto! (Digite 'sair' para encerrar)")
print("-" * 50)

# 4. Loop do Chat
while True:
    pergunta = input("\nVocê (Calouro): ")
    if pergunta.strip().lower() == 'sair':
        print("Até logo!")
        break

    print("🔎 A procurar nos documentos da URCA...")
    pergunta_lower = pergunta.lower()
    
    paginas_selecionadas = []
    fontes_selecionadas = []

    # ROTEAMENTO INTELIGENTE: Se falar de horário, professor, semestre, período, matriz ou disciplina,
    # garante que TODAS as páginas da Matriz e do Horário entrem sem falha!
    palavras_grade = ["horário", "horario", "horarios", "semestre", "semstre", "semetrte", "período", "periodo", "pereildo", "perildos", "matriz", "disciplina", "diciplina", "matéria", "materia", "professor", "professora", "grade", "ementa", "emenda", "cadeira", "optativa"]
    
    if any(p in pergunta_lower for p in palavras_grade):
        for texto, meta in zip(todos_os_dados["documents"], todos_os_dados["metadatas"]):
            nome_arq = meta.get("source", "").lower()
            if "horário" in nome_arq or "horario" in nome_arq or "matriz" in nome_arq:
                paginas_selecionadas.append(texto)
                fontes_selecionadas.append(f"{meta.get('source')} (Pág. {meta.get('page')})")

    # Complementa sempre com os 5 melhores resultados da busca semântica (para pegar Manual Acadêmico, POP e Atividades Complementares)
    resultados_busca = vector_store.similarity_search(pergunta, k=5)
    for doc in resultados_busca:
        if doc.page_content not in paginas_selecionadas:
            paginas_selecionadas.append(doc.page_content)
            fontes_selecionadas.append(f"{doc.metadata.get('source')} (Pág. {doc.metadata.get('page')})")

    print(f"📄 Páginas enviadas para leitura ({len(paginas_selecionadas)}): {', '.join(fontes_selecionadas)}")
    
    contexto = "\n\n".join(paginas_selecionadas)
    historico_texto = "\n".join(historico_conversa[-4:]) # Mantém as últimas 2 perguntas e respostas na memória

    prompt = f"""Você é o assistente virtual oficial da URCA (Universidade Regional do Cariri), focado no curso de Enfermagem.
    
    Regras importantes:
    1. Responda de forma amigável, organizada e direta, baseando-se APENAS nos documentos oficiais abaixo.
    2. Atenção: nos documentos, '1º Período', 'I Período', '1º Semestre' ou 'S1' significam a mesma coisa. Mesmo que o aluno escreva com erros de digitação (ex: 'pereildo', 'semetrte', 'infermagem', 'emenda'), entenda a intenção dele.
    3. Use o Histórico da Conversa para entender perguntas de continuação (como 'e os outros períodos?').

    Histórico recente da conversa:
    {historico_texto}

    Documentos oficiais da URCA:
    {contexto}
    
    Pergunta atual do aluno: {pergunta}"""

    respondeu = False
    for modelo_atual in modelos_para_tentar:
        try:
            resposta = client.models.generate_content(
                model=modelo_atual,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            print(f"\nBot URCA ({modelo_atual}):\n{resposta.text}")
            
            # Salva na memória da conversa
            historico_conversa.append(f"Aluno: {pergunta}")
            historico_conversa.append(f"Bot: {resposta.text}")
            
            respondeu = True
            break
            
        except Exception as e:
            erro_str = str(e)
            if "503" in erro_str or "UNAVAILABLE" in erro_str:
                print(f"⚠️ {modelo_atual} ocupado (503). Tentando próximo modelo...")
                time.sleep(1)
            elif "404" in erro_str or "NOT_FOUND" in erro_str or "400" in erro_str:
                continue
            else:
                print(f"⚠️ Falha no modelo {modelo_atual}: {e}")
                time.sleep(1)

    if not respondeu:
        print("\n❌ Nenhum modelo conseguiu responder agora. Tente novamente em alguns instantes.")