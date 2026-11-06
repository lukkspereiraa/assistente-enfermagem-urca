import os
import shutil
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PASTA_BANCO = "./chroma_db_urca"

if os.path.exists(PASTA_BANCO):
    print("🧹 A apagar o banco antigo...")
    shutil.rmtree(PASTA_BANCO)

arquivos_pdf = []
for raiz, _, arquivos in os.walk("."):
    for arquivo in arquivos:
        if arquivo.lower().endswith(".pdf"):
            arquivos_pdf.append(os.path.join(raiz, arquivo))

print(f"📚 PDFs encontrados ({len(arquivos_pdf)}): {[os.path.basename(f) for f in arquivos_pdf]}")

documentos_finais = []

for caminho_pdf in arquivos_pdf:
    nome_arquivo = os.path.basename(caminho_pdf)
    leitor = PdfReader(caminho_pdf)
    
    for num_pagina, pagina in enumerate(leitor.pages, start=1):
        texto_pagina = pagina.extract_text() or ""
        if not texto_pagina.strip():
            continue
            
        # Mantém a página INTEIRA junta para nunca cortar tabelas de horários ou matriz ao meio!
        texto_completo = f"=== DOCUMENTO: {nome_arquivo} (PÁGINA {num_pagina}) ===\n{texto_pagina}"
        
        doc = Document(
            page_content=texto_completo,
            metadata={"source": nome_arquivo, "page": num_pagina}
        )
        documentos_finais.append(doc)

print(f"📄 Total de páginas inteiras indexadas: {len(documentos_finais)}")

print("🧠 A gerar embeddings e salvar no ChromaDB...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Chroma.from_documents(
    documents=documentos_finais,
    embedding=embeddings_model,
    persist_directory=PASTA_BANCO
)

print("✅ Banco de páginas inteiras criado com sucesso!")