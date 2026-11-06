import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# A nossa base de conhecimento
arquivos_pdf = [
    "arquivos_pdf/Manual-Academico-2020.pdf",
    "arquivos_pdf/POP dos calouros.pdf_20250319_111013_0000.pdf",
    "arquivos_pdf/(2026.2) HORÁRIO DO CURSO DE ENFERMAGEM.pdf",
    "arquivos_pdf/matriz-enfermagem.pdf",
    "arquivos_pdf/Atividades complementares .pdf"
]

def extrair_texto(caminho):
    """Extrai texto de todas as páginas de um PDF."""
    texto = ""
    try:
        with fitz.open(caminho) as pdf:
            for pagina in pdf:
                # Extração simples de texto. 
                # (Se as tabelas ficarem desformatadas, podemos injetar uma heurística de leitura visual depois)
                texto += pagina.get_text("text") + "\n"
    except Exception as e:
        print(f"Erro ao ler {caminho}: {e}")
    return texto

# Configurar o divisor (1000 caracteres por bloco, 200 de sobreposição de segurança)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks_finais = []

# Processar cada ficheiro iterativamente
for arquivo in arquivos_pdf:
    conteudo = extrair_texto(arquivo)
    if conteudo:
        # Guardamos a "fonte" para que o bot possa citar a origem da resposta
        chunks = text_splitter.create_documents([conteudo], metadatas=[{"fonte": arquivo}])
        chunks_finais.extend(chunks)
        print(f"[{arquivo}] processado: {len(chunks)} blocos gerados.")

print(f"\nTotal de fragmentos prontos para a base vetorial: {len(chunks_finais)}")

# Exibir um exemplo prático do primeiro bloco gerado
if chunks_finais:
    print("\nExemplo do primeiro bloco:")
    print(chunks_finais[0].page_content)