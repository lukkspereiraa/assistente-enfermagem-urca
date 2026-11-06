import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { ChromaClient } from 'chromadb';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Iniciar o modelo Gemini
const llm = new ChatGoogleGenerativeAI({
    modelName: 'gemini-1.5-flash',
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 0.3 // Baixa temperatura para não inventar regras
});

// Ligar à base Chroma local criada em Python
const chroma = new ChromaClient({ path: "http://localhost:8000" });

app.post('/api/chat', async (req, res) => {
    try {
        const { pergunta } = req.body;

        // 1. Ir buscar o contexto à base de dados (pesquisa vetorial)
        const collection = await chroma.getCollection({ name: "langchain" }); 
        const resultados = await collection.query({
            queryTexts: [pergunta],
            nResults: 3
        });

        // 2. Juntar os textos encontrados num único bloco de contexto
        const contexto = resultados.documents[0].join("\n\n");

        // 3. Criar o prompt para o Gemini
        const prompt = `Você é o assistente virtual da URCA (Universidade Regional do Cariri), focado no curso de Enfermagem.
        Responda à pergunta do aluno de forma clara, amigável e direta, baseando-se APENAS no contexto abaixo.
        Se a informação não estiver no contexto, diga que não tem essa informação no momento.
        
        Contexto dos documentos oficiais:
        ${contexto}
        
        Pergunta do aluno: ${pergunta}
        
        Resposta:`;

        // 4. Pedir ao Gemini para gerar a resposta
        const resposta = await llm.invoke(prompt);

        res.json({ resposta: resposta.content });

    } catch (error) {
        console.error("Erro:", error);
        res.status(500).json({ erro: "Ocorreu um erro ao processar a pergunta." });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Servidor rodando na porta ${PORT} 🚀`);
});