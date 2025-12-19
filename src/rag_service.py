"""
RAG-сервис для инженерного чат-бота.
Обеспечивает поиск релевантных документов и генерацию ответов на основе найденного контекста.
Использует Hugging Face Router API (OpenAI-совместимый) для LLM и локальные embeddings.
Модель по умолчанию: Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langfuse import Langfuse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "engineer_bot"
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"

# рекомендуемые модели Hugging Face для русского языка (OpenAI-совместимый) :
# - Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud (vision-модель, отличная поддержка русского)
# - meta-llama/Llama-3.1-8B-Instruct (быстрее, средняя точность)
# - mistralai/Mistral-7B-Instruct-v0.2 (быстрая)

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud"

# системный промпт (без истории, она добавляется через MessagesPlaceholder)
SYSTEM_PROMPT = """
Ты профессиональный ассистент по приёмо-сдаточным и эксплуатационным испытаниям электроустановок. 
Твоя задача - отвечать на вопросы на основе предоставленного контекста 
из нормативной документации (ПУЭ, ПТЭЭП, СП, ГОСТ, РД) и примеров технических отчётов.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из нормативных документов и примеров отчётов
2. Если в контексте нет информации для ответа, честно скажи об этом
3. Всегда цитируй конкретные документы (название, номер, пункт/раздел) при ответе
4. Будь точным и конкретным, используй технические термины правильно
5. Если вопрос требует расчетов, приведи формулы и объясни шаги
6. Учитывай контекст предыдущих сообщений в диалоге, если это уместно
7. Внимательно анализируй ВСЕ предоставленные документы - нужная информация может быть в любом из них
8. При интерпретации результатов измерений сравнивай с нормативными требованиями и делай вывод о соответствии/несоответствии
9. При вопросах о протоколах - используй примеры из технических отчётов для иллюстрации
10. При вопросах о требованиях - ищи информацию в нормативных документах (ПУЭ, ПТЭЭП, СП, ГОСТ, РД)

Особенности работы:
- При вопросах о сопротивлении заземления - ищи в ПУЭ 1.8.39, РД 34.45-51.300-97
- При вопросах о сопротивлении изоляции - ищи в ПУЭ п.1.8.40
- При вопросах об автоматических выключателях - ищи в ПУЭ п.1.8.37
- При вопросах о цепи фаза-нуль - ищи в ПТЭЭП Прил. 3 п.28.4
- При вопросах о характеристиках оборудования - используй справочники и примеры из отчётов
- При вопросах о заполнении протоколов - используй примеры из технических отчётов

Контекст из документов:
{context}
"""

class RAGService:
    """Сервис для RAG-поиска и генерации ответов через Hugging Face Inference API"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        k: int = 5,  # увеличено с 3 до 5 для лучшего покрытия документов
        enable_langfuse: bool = True,
        hf_token: Optional[str] = None,
        enable_memory: bool = True
    ):
        """
        Инициализация RAG-сервиса.
        
        Args:
            model_name: Название модели Hugging Face (например, "mistralai/Mistral-7B-Instruct-v0.2")
            temperature: Температура для генерации (ниже = более детерминированно)
            k: Количество документов для извлечения
            enable_langfuse: Включить трейсинг через Langfuse
            hf_token: Hugging Face API токен (если None, берется из HUGGINGFACE_API_TOKEN)
            enable_memory: Включить историю диалога (memory)
        """
        self.k = k if k >= 3 else 5  # минимум 5 документов для лучшего покрытия (было 3)
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.temperature = temperature
        self.enable_langfuse = enable_langfuse
        self.enable_memory = enable_memory
        # поддерживаем оба варианта названия переменной
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN или HUGGINGFACE_API_TOKEN не найден в переменных окружения"
                "получите токен на https://huggingface.co/settings/tokens"
            )
        
        self.memory = None
        if self.enable_memory:
            self.memory = ChatMessageHistory()
            logger.info("история диалога (memory) включена")
        
        self.langfuse = None
        if self.enable_langfuse:
            self._init_langfuse()
        
        self._init_vectorstore()
        
        self._init_llm()

        self._init_chain()
    
    def _init_langfuse(self):
        """Инициализация Langfuse для трейсинга"""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if public_key and secret_key:
            try:
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                print("Langfuse инициализирован")
            except Exception as e:
                print(f"Не удалось инициализировать Langfuse: {e}")
                print("   Трейсинг будет отключен")
                self.langfuse = None
                self.enable_langfuse = False
        else:
            print("Langfuse ключи не найдены в переменных окружения")
            print("Трейсинг будет отключен. Нужно установить LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY")
            self.enable_langfuse = False
    
    def _init_vectorstore(self):
        """Инициализация векторной базы данных Chroma"""
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )
    
    def _init_llm(self):
        """Инициализация LLM через HF Router API (OpenAI-совместимый)"""
        try:
            # используем OpenAI-совместимый API через Hugging Face Router
            # аналогично client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                base_url="https://router.huggingface.co/v1",
                api_key=self.hf_token,
                max_retries=3,
                timeout=120,  # увеличен таймаут для больших моделей (72B)
                default_headers={
                    "x-use-cache": "false"  # отключили кеш для свежих ответов
                }
            )
            logger.info(f"LLM инициализирован: {self.model_name} (через HF Router)")
        except Exception as e:
            raise RuntimeError(
                f"Не удалось инициализировать LLM {self.model_name}: {e}\n"
                "Убедитесь, что:\n"
                "1. HF_TOKEN или HUGGINGFACE_API_TOKEN установлен и валиден\n"
                "2. Модель доступна через Hugging Face Router API\n"
                "3. У вас есть доступ к модели (может потребоваться принять условия использования)\n"
                "4. Для больших моделей (72B) может потребоваться больше времени на ответ"
            ) from e
    
    def _init_chain(self):
        """Инициализация RAG-цепочки"""
        if self.enable_memory and self.memory:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
        else:
            # если без истории диалога, то убираем {chat_history} из промпта
            system_prompt_no_history = SYSTEM_PROMPT.replace("{chat_history}\n\n", "")
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt_no_history),
                ("human", "{question}")
            ])
        
        # цепочка retriever-format_docs-prompt-llm-parser
        if self.enable_memory and self.memory:
            # используем RunnableLambda для получения истории диалога
            # функция игнорирует входные данные и просто возвращает историю из memory
            def get_chat_history(_):
                """Получает историю диалога из memory."""
                if self.memory:
                    return self.memory.messages
                return []
            
            chat_history_lambda = RunnableLambda(get_chat_history)
            
            from operator import itemgetter
            
            self.chain = (
                {
                    "context": itemgetter("question") | self.retriever | self._format_docs,
                    "question": itemgetter("question"),
                    "chat_history": chat_history_lambda
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Форматирует найденные документы в строку для контекста"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            name = doc.metadata.get("name", source)
            revision = doc.metadata.get("revision", "unknown")
            # конвертируем chunk_index в строку если он число
            chunk_idx = doc.metadata.get("chunk_index", "?")
            if isinstance(chunk_idx, int):
                chunk_idx = str(chunk_idx)
            elif chunk_idx is None:
                chunk_idx = "?"
            doc_type = doc.metadata.get("type", "unknown")
            formatted.append(
                f"[Документ {i}]\n"
                f"Название: {name}\n"
                f"Тип: {doc_type}\n"
                f"Ревизия: {revision}\n"
                f"Источник: {source} (чанк {chunk_idx})\n"
                f"Содержание:\n{doc.page_content}\n"
            )
        
        return "\n---\n".join(formatted)
    
    def _simplify_query(self, question: str) -> str:
        """
        Упрощает запрос для лучшего семантического поиска,
        удаляет служебные слова и фразы, оставляя ключевые термины
        """
        stop_phrases = [
            "список", "перечисли", "назови", "укажи", "приведи", "дай",
            "с их", "с указанием", "включая", "со следующими", "какие",
            "что такое", "расскажи", "объясни про"
        ]
        
        stop_words = ["и", "или", "для", "при", "с", "со", "их", "его", "её"]
        
        simplified = question.lower().strip()

        for phrase in stop_phrases:
            simplified = simplified.replace(phrase, "")
        
        words = simplified.split()

        if len(words) > 2:
            words = [w for w in words if w not in stop_words]
        
        simplified = " ".join(words)
        
        if len(simplified.split()) < 2:
            return question
        
        return simplified
    
    def ask(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Задает вопрос и получает ответ на основе RAG
        
        Args:
            question: Вопрос пользователя
            filters: Опциональные фильтры для поиска (например, {"type": "ГОСТ"})
            trace_id: Опциональный ID трейса для Langfuse (если None, создается новый)
        
        Returns:
            Словарь с ответом и источниками:
            {
                "answer": str,
                "sources": List[Dict],
                "question": str,
                "trace_id": str (если используется Langfuse)
            }
        """

        search_query = self._simplify_query(question)
        logger.debug(f"Оригинальный запрос: {question}, упрощенный: {search_query}")
        
        trace = None
        trace_id_used = None
        if self.enable_langfuse and self.langfuse:

            trace_context = None
            if trace_id:
                trace_context = {"trace_id": trace_id}
            
            trace = self.langfuse.start_span(
                name="engineer-rag",
                input={"question": question, "filters": filters},
                trace_context=trace_context
            )
            trace_id_used = trace.id if hasattr(trace, 'id') else None

        if filters:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k, "filter": filters}
            )
            self._init_chain()

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_span(
                name="retrieval",
                input={"original_query": question, "search_query": search_query, "k": self.k, "filters": filters}
            )
        
        retrieval_start = time.time()
        # используем упрощенный запрос для лучшего семантического поиска
        try:
            docs = self.retriever.invoke(search_query)
        except AttributeError:
            docs = self.retriever.get_relevant_documents(search_query)
        retrieval_time = time.time() - retrieval_start
        
        sources = []
        sources_metadata = []
        for doc in docs:
            chunk_idx = doc.metadata.get("chunk_index", "?")
            if isinstance(chunk_idx, int):
                chunk_idx = str(chunk_idx)
            elif chunk_idx is None:
                chunk_idx = "?"
            
            source_info = {
                "source": doc.metadata.get("source", "unknown"),
                "name": doc.metadata.get("name", doc.metadata.get("source", "unknown")),
                "revision": doc.metadata.get("revision", "unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "category": doc.metadata.get("category", "unknown"),
                "chunk_index": chunk_idx,
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
            sources_metadata.append({
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:200]
            })
        
        if retrieval_span:
            retrieval_span.update(
                output={
                    "documents_count": len(docs),
                    "sources": sources_metadata,
                    "retrieval_time_seconds": retrieval_time
                },
                metadata={
                    "k": self.k,
                    "filters": filters
                }
            )
            retrieval_span.end()
        
        generation_span = None
        if trace:
            context = self._format_docs(docs)
            generation_span = trace.start_span(
                name="generation",
                input={
                    "question": question,
                    "context_length": len(context),
                    "context_preview": context[:500] + "..." if len(context) > 500 else context
                }
            )
        
        generation_start = time.time()

        try:
            if self.enable_memory and self.memory:
                history_count = len(self.memory.messages) if self.memory else 0
                logger.debug(f"История диалога перед вызовом: {history_count} сообщений")
                if history_count > 0:
                    logger.debug(f"Последние сообщения: {[msg.content[:50] for msg in self.memory.messages[-2:]]}")
                
                answer = self.chain.invoke({"question": question})
            else:
                answer = self.chain.invoke(question)

            if self.enable_memory and self.memory:
                self.memory.add_user_message(question)
                self.memory.add_ai_message(answer)
                logger.debug(f"История сохранена. Всего сообщений в memory: {len(self.memory.messages)}")
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"
            logger.error(error_msg)
            if generation_span:
                generation_span.update(
                    output={"error": error_msg},
                    level="ERROR"
                )
                generation_span.end()
            if trace:
                trace.update(output={"error": error_msg}, level="ERROR")
                trace.end()
            raise RuntimeError(error_msg) from e
        
        generation_time = time.time() - generation_start
        
        if generation_span:
            generation_span.update(
                output={
                    "answer": answer,
                    "answer_length": len(answer),
                    "generation_time_seconds": generation_time
                },
                metadata={
                    "model": self.model_name,
                    "temperature": self.temperature
                }
            )
            generation_span.end()

        result = {
            "answer": answer,
            "sources": sources,
            "question": question
        }
        
        if trace:
            trace.update(
                output=result,
                metadata={
                    "total_time_seconds": retrieval_time + generation_time,
                    "retrieval_time_seconds": retrieval_time,
                    "generation_time_seconds": generation_time,
                    "sources_count": len(sources),
                    "model": self.model_name
                }
            )
            trace.end()
            if not trace_id_used:
                trace_id_used = trace.id if hasattr(trace, 'id') else None
            result["trace_id"] = trace_id_used
        
        if filters:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            )
            self._init_chain()
        
        return result
    
    def clear_memory(self):
        """Очищает историю диалога."""
        if self.memory:
            self.memory.clear()
            logger.info("История диалога очищена")
    
    def get_memory_history(self) -> List[Dict[str, str]]:
        """Возвращает историю диалога."""
        if not self.memory:
            return []
        
        history = []
        messages = self.memory.messages
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "question": messages[i].content if hasattr(messages[i], 'content') else str(messages[i]),
                    "answer": messages[i + 1].content if hasattr(messages[i + 1], 'content') else str(messages[i + 1])
                })
        return history
    
    def search_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Только поиск документов без генерации ответа.
        
        Args:
            query: Поисковый запрос
            k: Количество документов (если None, используется self.k)
        
        Returns:
            Список найденных документов
        """
        if k is not None:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            try:
                return retriever.invoke(query)
            except AttributeError:
                return retriever.get_relevant_documents(query)
        try:
            return self.retriever.invoke(query)
        except AttributeError:
            return self.retriever.get_relevant_documents(query)
    
    def add_score(
        self,
        trace_id: str,
        score_name: str = "quality",
        value: float = 1.0,
        comment: Optional[str] = None
    ):
        """
        Добавляет оценку (score) к существующему trace в Langfuse.
        
        Args:
            trace_id: ID трейса
            score_name: Название оценки (например, "quality", "relevance")
            value: Значение оценки (обычно 0.0-1.0 или 1-5)
            comment: Опциональный комментарий
        """
        if not self.enable_langfuse or not self.langfuse:
            print("Langfuse не инициализирован, оценка не добавлена")
            return
        
        try:
            self.langfuse.create_score(
                trace_id=trace_id,
                name=score_name,
                value=value,
                comment=comment,
                data_type="NUMERIC"  # по умолчанию числовая оценка
            )
            print(f"Оценка '{score_name}' = {value} добавлена к trace {trace_id}")
        except Exception as e:
            print(f"Ошибка при добавлении оценки: {e}")
            logger.error(f"Ошибка при добавлении оценки: {e}")


def create_rag_service(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    k: int = 5,  # Увеличено с 3 до 5 для лучшего покрытия
    enable_langfuse: bool = True,
    hf_token: Optional[str] = None,
    enable_memory: bool = True
) -> RAGService:
    """
    Создает и возвращает экземпляр RAGService.
    
    Args:
        model_name: Название модели Hugging Face
        temperature: Температура генерации
        k: Количество документов для извлечения
        enable_langfuse: Включить трейсинг через Langfuse
        hf_token: Hugging Face API токен
        enable_memory: Включить историю диалога
    
    Returns:
        Экземпляр RAGService
    """
    return RAGService(
        model_name=model_name,
        temperature=temperature,
        k=k,
        enable_langfuse=enable_langfuse,
        hf_token=hf_token,
        enable_memory=enable_memory
    )


if __name__ == "__main__":
    print("Инициализация RAG-сервиса...")
    
    model_name = os.getenv("HF_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    try:
        service = create_rag_service(model_name=model_name)
        print(f"RAG-сервис инициализирован (модель: {model_name})")
        
        question = "Какой класс бетона используется для фундаментов?"
        print(f"\nВопрос: {question}\n")
        
        result = service.ask(question)
        
        print(f"Ответ:\n{result['answer']}\n")
        print(f"Источники ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['name']} (ревизия: {source['revision']})")
            print(f"     {source['snippet'][:100]}...")
        
        if 'trace_id' in result:
            print(f"\nLangfuse Trace ID: {result['trace_id']}")
            print("   Можете добавить оценку через: service.add_score(trace_id, 'quality', 1.0)")
    
    except Exception as e:
        print(f"Ошибка: {e}")
        print("\nУбедитесь, что:")
        print("  1. Выполнен ingest.py для создания векторной базы")
        print("  2. HUGGINGFACE_API_TOKEN установлен (получите на https://huggingface.co/settings/tokens)")
        print("  3. Модель доступна через Inference API")
        print("  4. Установлены все зависимости из requirements.txt")
        import traceback
        traceback.print_exc()
