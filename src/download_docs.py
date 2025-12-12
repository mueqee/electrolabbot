"""
Скрипт для скачивания и подготовки открытых документов для базы знаний ЭлектроЛаббот.
Все документы - 100% легально и в открытом доступе.
"""

import os
import requests
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# База данных документов
DOCUMENTS_DB = [
    {
        "id": 1,
        "name": "ПУЭ-7 полная версия (актуальная редакция 2024)",
        "url": "https://sevcable.ru/files/pue_7.pdf",
        "filename": "pue_7.pdf",
        "size_mb": 35,
        "type": "normative",
        "category": "ПУЭ"
    },
    {
        "id": 2,
        "name": "ПТЭЭП (Правила технической эксплуатации электроустановок потребителей) 2023",
        "url": "http://docs.cntd.ru/document/1200035618",
        "filename": "pteepp_2023.pdf",
        "size_mb": None, 
        "type": "normative",
        "category": "ПТЭЭП"
    },
    {
        "id": 3,
        "name": "СП 256.1325800.2016 Электроустановки жилых и общественных зданий",
        "url": "http://docs.cntd.ru/document/456054197",
        "filename": "sp_256.1325800.2016.pdf",
        "size_mb": None,
        "type": "normative",
        "category": "СП"
    },
    {
        "id": 4,
        "name": "СП 484.1311500.2020 Системы пожарной сигнализации",
        "url": "http://docs.cntd.ru/document/566249686",
        "filename": "sp_484.1311500.2020.pdf",
        "size_mb": None,
        "type": "normative",
        "category": "СП"
    },
    {
        "id": 5,
        "name": "ГОСТ Р 50571.16-2019 (МЭК 60364-6:2016) — Периодические испытания",
        "url": "http://docs.cntd.ru/document/1200160939",
        "filename": "gost_r_50571.16-2019.pdf",
        "size_mb": None,
        "type": "normative",
        "category": "ГОСТ"
    },
    {
        "id": 6,
        "name": "РД 34.45-51.300-97 Объём и нормы испытаний электрооборудования",
        "url": "https://files.stroyinf.ru/Data1/46/46398.pdf",
        "filename": "rd_34.45-51.300-97.pdf",
        "size_mb": None,
        "type": "normative",
        "category": "РД"
    },
    # Примечание: Анонимизированные отчёты будут созданы отдельно
    # Справочники по оборудованию - пользователь должен скачать вручную
]


def download_file(url: str, filepath: Path, timeout: int = 300) -> bool:
    """
    Скачивает файл по URL.
    
    Args:
        url: URL файла
        filepath: Путь для сохранения
        timeout: Таймаут в секундах
        
    Returns:
        True если успешно, False иначе
    """
    try:
        logger.info(f"Скачивание {url} -> {filepath}")
        
        # Создаём директорию если её нет
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем файл
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Сохраняем файл
        total_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        size_mb = total_size / (1024 * 1024)
        logger.info(f"✓ Скачано: {filepath.name} ({size_mb:.2f} МБ)")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Ошибка при скачивании {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Неожиданная ошибка при скачивании {url}: {e}")
        return False


def download_all_documents(data_dir: Path, documents: List[Dict] = None) -> Dict[str, bool]:
    """
    Скачивает все документы из базы данных.
    
    Args:
        data_dir: Директория для сохранения документов
        documents: Список документов (если None, используется DOCUMENTS_DB)
        
    Returns:
        Словарь с результатами скачивания {filename: success}
    """
    if documents is None:
        documents = DOCUMENTS_DB
    
    results = {}
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Начинаю скачивание {len(documents)} документов в {raw_dir}")
    
    for doc in documents:
        filepath = raw_dir / doc["filename"]
        
        # Пропускаем если файл уже существует
        if filepath.exists():
            logger.info(f"⊘ Пропущено (уже существует): {doc['filename']}")
            results[doc["filename"]] = True
            continue
        
        success = download_file(doc["url"], filepath)
        results[doc["filename"]] = success
    
    # Статистика
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Скачивание завершено: {successful}/{total} успешно")
    logger.info(f"{'='*60}")
    
    return results


def create_documents_index(data_dir: Path) -> None:
    """
    Создаёт индексный файл со списком всех документов.
    
    Args:
        data_dir: Директория с документами
    """
    index_file = data_dir / "documents_index.md"
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("# Индекс документов базы знаний ЭлектроЛаббот\n\n")
        f.write("## Нормативные документы\n\n")
        f.write("| № | Документ | Файл | Размер | Категория |\n")
        f.write("|---|----------|------|--------|-----------|\n")
        
        for doc in DOCUMENTS_DB:
            size_str = f"{doc['size_mb']} МБ" if doc['size_mb'] else "—"
            f.write(f"| {doc['id']} | {doc['name']} | {doc['filename']} | {size_str} | {doc['category']} |\n")
        
        f.write("\n## Анонимизированные технические отчёты\n\n")
        f.write("15 полностью анонимизированных, но максимально реалистичных технических отчётов:\n\n")
        f.write("- Стадионы и манежи\n")
        f.write("- Торговые центры\n")
        f.write("- Школы и детские сады\n")
        f.write("- Производственные цеха\n")
        f.write("- Жилые дома\n\n")
        f.write("*(Отчёты будут добавлены в data/raw/reports/)*\n")
    
    logger.info(f"✓ Создан индекс документов: {index_file}")


if __name__ == "__main__":
    # Определяем директорию данных
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    logger.info("="*60)
    logger.info("ЭлектроЛаббот - Скачивание документов базы знаний")
    logger.info("="*60)
    logger.info(f"Директория данных: {data_dir}")
    logger.info("")
    
    # Скачиваем документы
    results = download_all_documents(data_dir)
    
    # Создаём индекс
    create_documents_index(data_dir)
    
    logger.info("\n✓ Готово! Теперь можно запустить ingest.py для индексации документов.")

