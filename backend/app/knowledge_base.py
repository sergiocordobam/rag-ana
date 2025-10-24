import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from typing import List, Dict, Tuple
import pickle

class AgriculturalKnowledgeBase:
    """
    Clase para manejar la base de conocimiento vectorial agrícola
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga y procesa los datos del archivo Excel
        """
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            return df
        except Exception as e:
            raise Exception(f"Error al cargar el archivo Excel: {str(e)}")
    
    def create_document_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convierte los datos del DataFrame en chunks de documentos para embeddings
        """
        documents = []
        
        # Crear chunks por fila con contexto
        for idx, row in df.iterrows():
            # Crear descripción de la fila
            row_text = f"Registro {idx + 1}: "
            
            # Agregar información de cada columna
            for col in df.columns:
                if pd.notna(row[col]):
                    row_text += f"{col}: {row[col]}. "
            
            documents.append({
                "text": row_text.strip(),
                "row_index": idx,
                "metadata": {
                    "row_data": row.to_dict(),
                    "columns": list(df.columns)
                }
            })
        
        # Crear chunks por columnas (análisis de tendencias)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Estadísticas numéricas
                stats = df[col].describe()
                stats_text = f"Análisis de {col}: "
                stats_text += f"Promedio: {stats['mean']:.2f}, "
                stats_text += f"Mínimo: {stats['min']:.2f}, "
                stats_text += f"Máximo: {stats['max']:.2f}, "
                stats_text += f"Desviación estándar: {stats['std']:.2f}"
                
                documents.append({
                    "text": stats_text,
                    "row_index": -1,  # Indica que es análisis estadístico
                    "metadata": {
                        "type": "statistical_analysis",
                        "column": col,
                        "statistics": stats.to_dict()
                    }
                })
        
        # Crear chunks de correlaciones
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Evitar duplicados
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.3:  # Solo correlaciones significativas
                            corr_text = f"Correlación entre {col1} y {col2}: {corr_value:.3f}"
                            documents.append({
                                "text": corr_text,
                                "row_index": -2,  # Indica que es análisis de correlación
                                "metadata": {
                                    "type": "correlation_analysis",
                                    "columns": [col1, col2],
                                    "correlation_value": corr_value
                                }
                            })
        
        print(f"✅ Creados {len(documents)} chunks de documentos")
        return documents
    
    def create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """
        Crea embeddings para todos los documentos
        """
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Embeddings creados: {embeddings.shape}")
        return embeddings
    
    def build_vector_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Construye el índice FAISS para búsqueda vectorial
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product para similitud coseno
        
        # Normalizar embeddings para similitud coseno
        faiss.normalize_L2(embeddings)
        
        # Agregar embeddings al índice
        self.index.add(embeddings.astype('float32'))
        
        # Guardar documentos y metadata
        self.documents = documents
        
        print(f"✅ Índice vectorial construido con {self.index.ntotal} documentos")
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Busca documentos similares a la consulta
        """
        if self.index is None:
            raise Exception("El índice vectorial no ha sido construido")
        
        # Crear embedding de la consulta
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Buscar documentos similares
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "similarity_score": float(score),
                    "rank": len(results) + 1
                })
        
        return results
    
    def initialize_from_excel(self, file_path: str):
        """
        Inicializa la base de conocimiento desde un archivo Excel
        """
        print("🔄 Inicializando base de conocimiento...")
        
        # Cargar datos
        df = self.load_excel_data(file_path)
        
        # Crear documentos
        documents = self.create_document_chunks(df)
        
        # Crear embeddings
        embeddings = self.create_embeddings(documents)
        
        # Construir índice
        self.build_vector_index(embeddings, documents)
        
        print("✅ Base de conocimiento inicializada correctamente")
    
    def save_knowledge_base(self, file_path: str):
        """
        Guarda la base de conocimiento en disco
        """
        if self.index is None:
            raise Exception("No hay base de conocimiento para guardar")
        
        # Guardar índice FAISS
        faiss.write_index(self.index, f"{file_path}.index")
        
        # Guardar documentos y metadata
        with open(f"{file_path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ Base de conocimiento guardada en {file_path}")
    
    def load_knowledge_base(self, file_path: str):
        """
        Carga la base de conocimiento desde disco
        """
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(f"{file_path}.index")
            
            # Cargar documentos
            with open(f"{file_path}.docs", 'rb') as f:
                self.documents = pickle.load(f)
            
            print(f"✅ Base de conocimiento cargada desde {file_path}")
        except Exception as e:
            raise Exception(f"Error al cargar la base de conocimiento: {str(e)}")
