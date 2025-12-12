"""
Скрипт для индексации документов в векторную базу данных Chroma.
Читает документы из data/raw (PDF, TXT, DOCX, MD), разбивает на чанки, 
генерирует embeddings и сохраняет в Chroma.
"""

import os
import yaml
import tiktoken
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback для старых версий
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Конфигурация
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"  
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "engineer_bot"

# Параметры чанкинга
CHUNK_SIZE = 500  # токенов
CHUNK_OVERLAP = 75  # 15% от 500

# Поддерживаемые форматы
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md'}


def load_metadata(file_path: Path) -> Dict[str, Any]:
    """Загружает метаданные из YAML файла."""
    # Пробуем найти метаданные в processed директории
    metadata_file = PROCESSED_DIR / f"{file_path.stem}.metadata.yaml"
    if not metadata_file.exists():
        # Пробуем в той же директории
        metadata_file = file_path.parent / f"{file_path.stem}.metadata.yaml"
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Ошибка при загрузке метаданных {metadata_file}: {e}")
    
    return {}


def count_tokens(text: str) -> int:
    """Подсчитывает количество токенов в тексте (приблизительно)."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def parse_pdf(file_path: Path) -> str:
    """Парсит PDF файл и возвращает текст."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Ошибка при парсинге PDF {file_path}: {e}")
        raise


def parse_docx(file_path: Path) -> str:
    """Парсит DOCX файл и возвращает текст."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Ошибка при парсинге DOCX {file_path}: {e}")
        raise


def parse_txt(file_path: Path) -> str:
    """Парсит TXT файл и возвращает текст."""
    try:
        # Пробуем разные кодировки
        encodings = ['utf-8', 'cp1251', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Не удалось декодировать файл {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при парсинге TXT {file_path}: {e}")
        raise


def parse_markdown(file_path: Path) -> str:
    """Парсит Markdown файл и возвращает текст."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка при парсинге Markdown {file_path}: {e}")
        raise


def load_document_content(file_path: Path) -> str:
    """Загружает содержимое документа в зависимости от его типа."""
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return parse_pdf(file_path)
    elif suffix == '.docx':
        return parse_docx(file_path)
    elif suffix == '.txt':
        return parse_txt(file_path)
    elif suffix == '.md':
        return parse_markdown(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def load_documents() -> List[Document]:
    """Загружает все документы из data/raw с метаданными."""
    documents = []
    
    if not DATA_DIR.exists():
        logger.warning(f"Директория {DATA_DIR} не существует. Создаю...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return documents
    
    # Находим все поддерживаемые файлы (рекурсивно, включая поддиректории)
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        # Ищем в корне и во всех поддиректориях
        all_files.extend(list(DATA_DIR.rglob(f"*{ext}")))
    
    if not all_files:
        logger.warning(f"Не найдено документов в {DATA_DIR}")
        return documents
    
    logger.info(f"Найдено {len(all_files)} документов для обработки")
    
    for file_path in all_files:
        try:
            # Пропускаем файлы метаданных
            if file_path.name.endswith('.metadata.yaml'):
                continue
            
            logger.info(f"Обработка файла: {file_path.name}")
            
            # Загружаем содержимое
            content = load_document_content(file_path)
            
            if not content.strip():
                logger.warning(f"Файл {file_path.name} пуст, пропускаю")
                continue
            
            # Загружаем метаданные
            metadata = load_metadata(file_path)
            
            # Создаем базовый документ
            doc_metadata = {
                "source": file_path.name,
                "file_type": file_path.suffix.lower(),
                "type": metadata.get("type", "unknown"),
                "revision": metadata.get("revision", "unknown"),
                "date": metadata.get("date", "unknown"),
                "category": metadata.get("category", "unknown"),
                "name": metadata.get("name", file_path.stem),
            }
            
            # Добавляем теги, если есть
            if "tags" in metadata:
                doc_metadata["tags"] = ", ".join(metadata["tags"])
            
            documents.append(Document(page_content=content, metadata=doc_metadata))
            logger.info(f"Загружен документ: {file_path.name} ({len(content)} символов)")
        
        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path.name}: {e}")
            continue
    
    return documents


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Разбивает документы на чанки с учетом размера в токенах.
    
    Args:
        documents: Список документов для разбиения
        chunk_size: Размер чанка в токенах
        chunk_overlap: Перекрытие чанков в токенах
    """
    
    # Используем RecursiveCharacterTextSplitter
    # Он пытается сохранить структуру документа (параграфы, предложения)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    
    for doc in documents:
        try:
            # Разбиваем документ на чанки
            chunks = text_splitter.split_documents([doc])
            
            # Добавляем информацию о номере чанка
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            
            chunked_docs.extend(chunks)
        except Exception as e:
            logger.error(f"Ошибка при разбиении документа {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    
    return chunked_docs


def create_embeddings(use_api: bool = False):
    """
    Создает модель для генерации embeddings.
    
    Args:
        use_api: Если True, использует Hugging Face Inference API (требует HF_TOKEN)
                 Если False, использует локальную модель (бесплатно, быстрее)
    """
    if use_api:
        # Использование Hugging Face Inference API (требует HF_TOKEN)
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            print("HUGGINGFACE_API_TOKEN не найден, используем локальную модель")
            use_api = False
    
    if use_api:
        # Используем Inference API для embeddings
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # Локальная модель - бесплатно и быстрее для индексации
        # Используем ruBERT для лучшей работы с русским языком
        model_name = "cointegrated/rubert-tiny2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Используем CPU, можно изменить на 'cuda' если есть GPU
            encode_kwargs={'normalize_embeddings': True}  # Нормализация для cosine similarity
        )
    
    return embeddings


def ingest_documents(chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
    """
    Основная функция индексации документов.
    
    Args:
        chunk_size: Размер чанка в токенах (по умолчанию CHUNK_SIZE)
        chunk_overlap: Перекрытие чанков в токенах (по умолчанию CHUNK_OVERLAP)
    """
    logger.info("🚀 Начало индексации документов...")
    
    # Используем переданные параметры или значения по умолчанию
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    # 1. Загружаем документы
    logger.info(f"Загрузка документов из {DATA_DIR}...")
    documents = load_documents()
    
    if not documents:
        logger.error("Не найдено документов для индексации")
        logger.info(f"Поместите документы (PDF, TXT, DOCX, MD) в директорию {DATA_DIR}")
        return None
    
    logger.info(f"Загружено {len(documents)} документов")
    
    # 2. Разбиваем на чанки
    logger.info(f"Разбиение на чанки (размер: {chunk_size} токенов, overlap: {chunk_overlap})...")
    try:
        chunked_docs = chunk_documents(documents, chunk_size, chunk_overlap)
        logger.info(f"Создано {len(chunked_docs)} чанков")
    except Exception as e:
        logger.error(f"Ошибка при разбиении на чанки: {e}")
        raise
    
    # 3. Создаем модель embeddings
    logger.info("Загрузка модели для embeddings...")
    try:
        embeddings = create_embeddings()
        logger.info("Модель загружена")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели embeddings: {e}")
        raise
    
    # 4. Создаем или обновляем векторную базу
    logger.info(f"Сохранение в Chroma (коллекция: {COLLECTION_NAME})...")
    
    try:
        # Удаляем старую базу, если существует
        if CHROMA_PERSIST_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_PERSIST_DIR)
            logger.info("Удалена старая база данных")
        
        # Создаем новую базу
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
            collection_name=COLLECTION_NAME
        )
        
        logger.info(f"Документы проиндексированы и сохранены в {CHROMA_PERSIST_DIR}")
        
        # 5. Проверяем количество документов в базе
        try:
            # Пытаемся получить количество через публичный API
            if hasattr(vectorstore, '_collection'):
                count = vectorstore._collection.count()
            else:
                # Альтернативный способ - через retriever
                count = len(chunked_docs)
            logger.info(f"Всего векторов в базе: {count}")
        except Exception as e:
            logger.warning(f"Не удалось получить точное количество векторов: {e}")
            logger.info(f"Индексировано документов: {len(chunked_docs)}")
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении в Chroma: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Индексация документов в векторную базу")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Размер чанка в токенах")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Перекрытие чанков в токенах")
    
    args = parser.parse_args()
    
    try:
        vectorstore = ingest_documents(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        if vectorstore:
            logger.info("\nИндексация завершена успешно!")
            logger.info(f"База данных: {CHROMA_PERSIST_DIR}")
            logger.info(f"Коллекция: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"\nОшибка при индексации: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

