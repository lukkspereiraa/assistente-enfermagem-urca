import os
from dotenv import load_dotenv
from google import genai

# Carregar a chave
load_dotenv()
chave_gemini = os.getenv("GEMINI_API_KEY")

if not chave_gemini:
    print("❌ A chave não foi encontrada no ficheiro .env!")
    exit()

print("⏳ A testar a ligação aos servidores da Google...\n")

try:
    # Ligar o cliente
    client = genai.Client(api_key=chave_gemini)
    
    # Pedir a lista de modelos disponíveis
    print("✅ Ligação estabelecida! Modelos disponíveis para a sua chave:\n")
    
    # O SDK vai iterar sobre os modelos e mostrar o nome de cada um
    for model in client.models.list():
        # Vamos filtrar para mostrar apenas os que contêm 'gemini' para facilitar a leitura
        if 'gemini' in model.name:
            print(f"- {model.name}")
            
    print("\nTeste concluído com sucesso!")

except Exception as e:
    print(f"❌ Falha ao comunicar com a API: {e}")