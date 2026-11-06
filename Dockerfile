FROM python:3.11-slim

WORKDIR /app

# Copia e instala as dependências primeiro (aproveita o cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos do projeto (api.py, indexacao.py, index.html e os 5 PDFs)
COPY . .

# Cria a pasta de cache com permissão total para o Hugging Face baixar o modelo MiniLM
ENV HF_HOME=/tmp/huggingface

# Roda o indexacao.py durante o build para criar o banco chroma_db_urca limpo no servidor
RUN python indexacao.py

# Expõe a porta padrão do Hugging Face Spaces e inicia o FastAPI
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]