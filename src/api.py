"""
FastAPI приложение для RAG-бота.
Предоставляет REST API для задавания вопросов и получения ответов.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

from rag_service import create_rag_service, RAGService

# Загружаем переменные окружения
load_dotenv()

# Инициализация FastAPI
app = FastAPI(
    title="RAG HF Bot API",
    description="API для инженерного RAG-бота на базе Hugging Face Inference API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный экземпляр RAG-сервиса
rag_service: Optional[RAGService] = None


# Pydantic модели для запросов и ответов
class QuestionRequest(BaseModel):
    """Модель запроса вопроса."""
    question: str = Field(..., description="Вопрос пользователя", min_length=1, max_length=1000)
    filters: Optional[Dict[str, Any]] = Field(None, description="Фильтры для поиска документов")
    trace_id: Optional[str] = Field(None, description="ID трейса Langfuse (опционально)")


class SourceResponse(BaseModel):
    """Модель источника документа."""
    source: str
    name: str
    revision: str
    type: str
    category: str
    chunk_index: str
    snippet: str


class QuestionResponse(BaseModel):
    """Модель ответа на вопрос."""
    answer: str
    sources: list[SourceResponse]
    question: str
    trace_id: Optional[str] = None


class ScoreRequest(BaseModel):
    """Модель запроса для добавления оценки."""
    trace_id: str = Field(..., description="ID трейса")
    score_name: str = Field("quality", description="Название оценки")
    value: float = Field(..., description="Значение оценки (0.0-1.0 или 1-5)", ge=0, le=5)
    comment: Optional[str] = Field(None, description="Комментарий к оценке")


@app.on_event("startup")
async def startup_event():
    """Инициализация RAG-сервиса при запуске приложения."""
    global rag_service
    try:
        model_name = os.getenv("HF_LLM_MODEL")
        enable_langfuse = os.getenv("ENABLE_LANGFUSE", "true").lower() == "true"
        
        rag_service = create_rag_service(
            model_name=model_name,
            enable_langfuse=enable_langfuse
        )
        print("✅ RAG-сервис инициализирован")
    except Exception as e:
        print(f"❌ Ошибка при инициализации RAG-сервиса: {e}")
        raise


@app.get("/")
async def root():
    """Корневой endpoint."""
    return {
        "message": "RAG HF Bot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    return {
        "status": "healthy",
        "model": rag_service.model_name,
        "vector_db": "Chroma",
        "langfuse_enabled": rag_service.enable_langfuse
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Задает вопрос и получает ответ на основе RAG.
    
    Args:
        request: Запрос с вопросом и опциональными фильтрами
    
    Returns:
        Ответ с текстом, источниками и trace_id
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    try:
        result = rag_service.ask(
            question=request.question,
            filters=request.filters,
            trace_id=request.trace_id
        )
        
        # Преобразуем источники в Pydantic модели
        sources = [
            SourceResponse(**source) for source in result["sources"]
        ]
        
        return QuestionResponse(
            answer=result["answer"],
            sources=sources,
            question=result["question"],
            trace_id=result.get("trace_id")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.post("/score")
async def add_score(request: ScoreRequest):
    """
    Добавляет оценку к существующему trace в Langfuse.
    
    Args:
        request: Запрос с trace_id, названием оценки и значением
    
    Returns:
        Подтверждение добавления оценки
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    if not rag_service.enable_langfuse:
        raise HTTPException(status_code=400, detail="Langfuse не включен")
    
    try:
        rag_service.add_score(
            trace_id=request.trace_id,
            score_name=request.score_name,
            value=request.value,
            comment=request.comment
        )
        
        return {
            "status": "success",
            "message": f"Оценка '{request.score_name}' = {request.value} добавлена к trace {request.trace_id}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении оценки: {str(e)}")


@app.get("/search")
async def search_documents(query: str, k: Optional[int] = None):
    """
    Поиск документов без генерации ответа.
    
    Args:
        query: Поисковый запрос
        k: Количество документов (по умолчанию используется настройка сервиса)
    
    Returns:
        Список найденных документов
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    try:
        docs = rag_service.search_only(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

