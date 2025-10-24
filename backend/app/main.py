from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
import json
from typing import Dict, Any, Optional
import ollama
from datetime import datetime
import uuid
from pydantic import BaseModel
from knowledge_base import AgriculturalKnowledgeBase

app = FastAPI(
    title="Agricultural RAG Analytics API",
    description="API para análisis de datos agrícolas usando RAG con Ollama",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Ollama
OLLAMA_MODEL = "mistral"  # Puedes cambiar este modelo según tu configuración

# Base de conocimiento global
knowledge_base = None
KNOWLEDGE_BASE_PATH = "knowledge_base"
EXCEL_FILE_PATH = "C:/Users/gmupe/Documents/rag-ana/backend/reporte_global.xlsx"  # Archivo fijo de base de conocimiento

# Modelos Pydantic
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

@app.get("/")
async def root():
    return {"message": "Agricultural RAG Analytics API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """
    Endpoint para hacer preguntas sobre los datos agrícolas usando RAG
    """
    global knowledge_base
    
    try:
        # Verificar que la base de conocimiento esté inicializada
        if knowledge_base is None:
            # Intentar cargar desde disco
            try:
                knowledge_base = AgriculturalKnowledgeBase()
                knowledge_base.load_knowledge_base(KNOWLEDGE_BASE_PATH)
                print("✅ Base de conocimiento cargada desde disco")
            except:
                # Si no existe, inicializar desde el archivo Excel fijo
                try:
                    print("🔄 Inicializando base de conocimiento desde reporte_global.xlsx...")
                    knowledge_base = AgriculturalKnowledgeBase()
                    knowledge_base.initialize_from_excel(EXCEL_FILE_PATH)
                    knowledge_base.save_knowledge_base(KNOWLEDGE_BASE_PATH)
                    print("✅ Base de conocimiento inicializada y guardada")
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error al inicializar la base de conocimiento desde {EXCEL_FILE_PATH}: {str(e)}"
                    )
        
        # Buscar documentos relevantes
        relevant_docs = knowledge_base.search_similar_documents(question, top_k=5)
        
        # Preparar contexto para Ollama
        context = "Información relevante de la base de conocimiento:\n\n"
        sources = []
        
        for i, doc_result in enumerate(relevant_docs):
            doc = doc_result["document"]
            score = doc_result["similarity_score"]
            
            context += f"{i+1}. {doc['text']}\n"
            sources.append({
                "text": doc["text"],
                "similarity_score": score,
                "metadata": doc["metadata"]
            })
        
        # Generar respuesta usando Ollama con RAG
        answer = await generate_rag_response(question, context)
        
        # Calcular confianza basada en similitud de documentos
        avg_confidence = sum([doc["similarity_score"] for doc in relevant_docs]) / len(relevant_docs) if relevant_docs else 0
        
        return JSONResponse(content={
            "answer": answer,
            "sources": sources,
            "confidence": round(avg_confidence, 3),
            "question": question,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al procesar la pregunta: {str(e)}"}
        )

async def generate_rag_response(question: str, context: str) -> str:
    """
    Genera respuesta usando Ollama con contexto RAG
    """
    try:
        prompt = f"""
Eres un experto en análisis de datos agrícolas. Basándote ÚNICAMENTE en la información proporcionada a continuación, responde la pregunta del usuario.

CONTEXTO:
{context}

PREGUNTA DEL USUARIO: {question}

INSTRUCCIONES:
1. Responde SOLO basándote en la información del contexto proporcionado
2. Si la información no es suficiente para responder completamente, indícalo claramente
3. Proporciona análisis detallados, predicciones y explicaciones de causas cuando sea posible
4. Incluye insights específicos basados en los datos
5. Responde en español
6. Sé específico y técnico en tu análisis agrícola

RESPUESTA:
"""
        
        # Llamar a Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': 'Eres un experto en análisis de datos agrícolas. Proporciona análisis detallados, predicciones y recomendaciones basadas únicamente en los datos proporcionados.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        return response['message']['content']
        
    except Exception as e:
        return f"Error al generar respuesta con Ollama: {str(e)}"


if __name__ == "__main__":
    try:
        import uvicorn
        print("🚀 Iniciando servidor FastAPI...")
        print("📖 Documentación disponible en: http://localhost:8000/docs")
        print("🛑 Presiona Ctrl+C para detener el servidor")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("❌ Error: uvicorn no está instalado")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
        print("💡 O usa: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")