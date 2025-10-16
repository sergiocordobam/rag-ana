from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
import json
from typing import Dict, Any
import ollama
import uuid

app = FastAPI(
    title="Agricultural Data Analytics API",
    description="API para análisis de datos agrícolas usando Ollama",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_MODEL = "mistral"

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Endpoint para recibir archivos Excel con información agrícola
    """
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="El archivo debe ser un archivo Excel (.xlsx o .xls)"
            )
        
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1]
        filename = f"{file_id}.{file_extension}"
        file_path = os.path.join("uploads", filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Error al leer el archivo Excel: {str(e)}"
            )
        
        analysis = await generate_analisis(df, file.filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if "error" in analysis:
            return JSONResponse(
                status_code=500,
                content={"error": analysis["error"]}
            )
        
        return JSONResponse(content={
            "analysis": analysis["analisis"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error interno del servidor: {str(e)}"}
        )

async def generate_analisis(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Genera análisis agrícola usando Ollama
    """
    try:
        data_summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(5).to_dict('records') if len(df) > 0 else [],
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        prompt = f"""
        Analiza los siguientes datos agrícolas del archivo '{filename}':

        Resumen de datos:
        - Total de filas: {data_summary['total_rows']}
        - Columnas: {', '.join(data_summary['columns'])}
        - Tipos de datos: {data_summary['data_types']}
        - Valores faltantes: {data_summary['missing_values']}

        Datos de muestra:
        {json.dumps(data_summary['sample_data'], indent=2)}

        Por favor proporciona un análisis detallado que incluya:
        1. Descripción general de los datos
        2. Patrones identificados
        3. Recomendaciones agrícolas basadas en los datos
        4. Posibles mejoras o consideraciones
        5. Insights clave para la agricultura

        Responde en español y sé específico sobre los datos proporcionados.
        """
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': 'Eres un experto en análisis de datos agrícolas. Proporciona análisis detallados y recomendaciones prácticas basadas en los datos proporcionados.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        return {
            "analisis": response['message']['content'],
        }
        
    except Exception as e:
        return {
            "error": f"Error al generar análisis con Ollama: {str(e)}"
        }
