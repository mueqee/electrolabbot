# RAG-ассистент по электротехническим отчётам и нормам

Открытый RAG-ассистент по приёмо-сдаточным и эксплуатационным испытаниям электроустановок. Работает на общедоступных документах и информации.

## Особенности

- **Открытая база знаний** - все документы легально в открытом доступе (ПУЭ, ПТЭЭП, СП, ГОСТ, РД)
- **Реальные примеры** - 15 анонимизированных технических отчётов с настоящими таблицами и протоколами
- **Бесплатный стек** - Hugging Face Inference API (30K запросов/месяц), Chroma (локально), Langfuse (бесплатный tier)
- **Русскоязычная поддержка** - ruBERT для embeddings, адаптирован для работы с русскоязычными документами
- **Профессиональные use cases** - поиск норм, интерпретация результатов, помощь в заполнении протоколов
- **Быстрый поиск** - локальное векторное хранилище Chroma с HNSW индексом
- **Наблюдаемость** - интеграция с Langfuse для трейсинга и анализа
- **Множество интерфейсов** - Streamlit UI, REST API (FastAPI) и CLI
- **Поддержка форматов** - PDF, TXT, DOCX, Markdown
- **История диалога** - память о предыдущих сообщениях в разговоре
- **Docker готовность** - полная поддержка Docker и docker-compose

## База знаний

### Нормативные документы (все в открытом доступе):

| № | Документ | Источник | Размер |
|---|----------|----------|--------|
| 1 | ПУЭ-7 полная версия (актуальная редакция 2024) | https://sevcable.ru/files/pue_7.pdf | ~35 МБ |
| 2 | ПТЭЭП (Правила технической эксплуатации электроустановок потребителей) 2023 | http://docs.cntd.ru/document/1200035618 | PDF |
| 3 | СП 256.1325800.2016 Электроустановки жилых и общественных зданий | http://docs.cntd.ru/document/456054197 | PDF |
| 4 | СП 484.1311500.2020 Системы пожарной сигнализации | http://docs.cntd.ru/document/566249686 | PDF |
| 5 | ГОСТ Р 50571.16-2019 (МЭК 60364-6:2016) — Периодические испытания | http://docs.cntd.ru/document/1200160939 | PDF |
| 6 | РД 34.45-51.300-97 «Объём и нормы испытаний электрооборудования» | https://files.stroyinf.ru/Data1/46/46398.pdf | PDF |

### Анонимизированные технические отчёты (15 штук):

Все отчёты созданы автоматически с использованием скрипта `src/generate_reports.py` и содержат реалистичные данные на основе типовых форм протоколов.

- **Стадионы и манежи** (3 отчёта)
  - Футбольный манеж
  - Стадион «Спорт», западная трибуна
  - Спортивный комплекс «Атлет»
  
- **Торговые центры** (3 отчёта)
  - Торговый центр «Центральный»
  - Торговый центр «Мега», подземная парковка
  - Торговый центр «Торг», складской комплекс
  
- **Школы и детские сады** (3 отчёта)
  - Средняя общеобразовательная школа № 1
  - Детский сад «Солнышко»
  - Средняя школа № 5, спортивный зал
  
- **Производственные цеха** (3 отчёта)
  - Производственный цех № 1
  - Склад готовой продукции
  - Административное здание завода
  
- **Жилые дома** (3 отчёта)
  - Многоквартирный жилой дом № 15
  - Жилой дом «Комфорт», подземная парковка
  - Жилой комплекс «Новый квартал»

Все отчёты содержат:
- Протоколы проверки наличия цепи между заземлёнными установками (ПУЭ 1.8.39)
- Протоколы проверки сопротивления изоляции (ПУЭ п.1.8.40)
- Протоколы проверки автоматических выключателей (ПУЭ п.1.8.37)
- Протоколы согласования параметров цепи «фаза – нуль» (ПТЭЭП Прил. 3 п.28.4)
- Заключения по результатам работ

**Расположение отчётов:** `data/raw/reports/`  
**Формат:** Markdown (.md)  
**Генерация:** `python src/generate_reports.py`

**Общий объём базы знаний:** ~300–400 МБ

## Структура проекта

```
electrolabbot/
├── configs/
│   ├── langfuse.yaml        # Параметры подключения к Langfuse
│   └── logging.yaml         # Настройки логгера
├── data/
│   ├── raw/                 # Исходные документы (PDF/MD/HTML)
│   └── processed/           # Конвертированные тексты и метаданные
├── notebooks/
│   └── .gitkeep             # Jupyter/Colab эксперименты
├── src/
│   ├── __init__.py
│   ├── ingest.py            # Скрипт индексации документов
│   ├── rag_service.py       # RAG-пайплайн + Langfuse трейсинг + Memory
│   ├── api.py               # FastAPI REST API
│   └── cli.py               # CLI-интерфейс
├── chroma_db/               # Векторная база данных (создается автоматически)
├── streamlit_app.py         # Streamlit UI приложение
├── Dockerfile               # Docker образ
├── docker-compose.yml       # Docker Compose конфигурация
├── .env.example             # Пример переменных окружения
├── requirements.txt         # Зависимости проекта
├── ARCHITECTURE.md          
└── README.md               
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Получение Hugging Face API токена

1. Зарегистрируйтесь на [Hugging Face](https://huggingface.co/join)
2. Перейдите в [Settings, Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте новый токен с правами `read`
4. Скопируйте токен

### 3. Настройка переменных окружения

```bash
cp .env.example .env
```

Отредактируйте `.env` и укажите:

```env
# Hugging Face API (обязательно)
HUGGINGFACE_API_TOKEN=your_hf_token_here

# Модель LLM (опционально, по умолчанию используется Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud)                                                                        
HF_LLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud

# Langfuse (опционально, для трейсинга)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# FastAPI (опционально)
PORT=8000
ENABLE_LANGFUSE=true
```

### 4. Генерация анонимизированных отчётов (опционально)

Для генерации 15 анонимизированных технических отчётов на основе имеющихся выполните:

```bash
python src/generate_reports.py
```

Отчёты будут созданы в `data/raw/reports/` в формате Markdown.

**Примечание:** Если вы хотите использовать свои отчёты, просто поместите их в `data/raw/reports/` или `data/raw/`.

### 5. Индексация документов

Поместите документы в `data/raw/` (форматы: **PDF, TXT, DOCX, Markdown**) и выполните:

```bash
python src/ingest.py
```

**Поддерживаемые форматы:**
- PDF (`.pdf`) — через pypdf
- TXT (`.txt`) — текстовые файлы с автоопределением кодировки
- DOCX (`.docx`) — документы Word через python-docx
- Markdown (`.md`) — markdown файлы

**Параметры индексации:**
```bash
python src/ingest.py --chunk-size 500 --chunk-overlap 75
```

Скрипт:
- Загрузит документы из `data/raw/`
- Разобьет их на чанки (по умолчанию 500 токенов, overlap 15%)
- Сгенерирует embeddings через ruBERT
- Сохранит в Chroma векторную базу

**Примеры документов** уже включены в `data/raw/` для тестирования.

### 6. Запуск

#### Streamlit UI (рекомендуется)

```bash
streamlit run streamlit_app.py
```

Приложение будет доступно на `http://localhost:8501`

**Возможности UI:**
- Интерактивный чат с историей диалога
- Автоматическое отображение источников
- Управление историей диалога
- Настройки в боковой панели

#### CLI (командная строка)

```bash
# Одиночный вопрос
python src/cli.py "Какой класс бетона используется для фундаментов?"

# Интерактивный режим
python src/cli.py --interactive

# С указанием модели
python src/cli.py "Вопрос" --model "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud"

# Без истории диалога
python src/cli.py "Вопрос" --no-langfuse
```

#### REST API

```bash
# Запуск сервера
python src/api.py

# Или через uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API будет доступен на `http://localhost:8000`

**Пример запроса:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Какой класс бетона используется для фундаментов?"}'
```

**Пример ответа:**

```json
{
  "answer": "Для фундаментов обычно используются классы бетона В20-В25...",
  "sources": [
    {
      "source": "gost_beton.md",
      "name": "ГОСТ 26633-2015. Бетоны тяжелые и мелкозернистые",
      "revision": "2015",
      "type": "ГОСТ",
      "category": "Бетон и железобетон",
      "chunk_index": "0",
      "snippet": "..."
    }
  ],
  "question": "Какой класс бетона используется для фундаментов?",
  "trace_id": "trace_123..."
}
```

## Стек

- **RAG Framework**: LangChain
- **LLM**: Hugging Face Router API (`Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud` или другие), OpenAI совместимо
- **Embeddings**: ruBERT (`cointegrated/rubert-tiny2`) - локально
- **Vector DB**: Chroma (локально, HNSW индекс)
- **Memory**: ConversationBufferMemory (история диалога)
- **Monitoring**: Langfuse (бесплатный tier)
- **UI**: Streamlit
- **API**: FastAPI
- **CLI**: Rich (красивый вывод в терминале)
- **Document Processing**: pypdf, python-docx, unstructured
- **Containerization**: Docker + docker-compose

## Рекомендуемые модели Hugging Face

### Для русского языка:

- `Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud` — vision-модель, отличная поддержка русского (по умолчанию)
- `mistralai/Mistral-7B-Instruct-v0.2` — хорошо работает с русским, быстрая
- `meta-llama/Llama-3.1-8B-Instruct` — отличная поддержка русского, вывод и точность
- `meta-llama/Llama-2-7b-chat-hf` — базовая поддержка русского, легкая и быстрая

### Для английского:

- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`
- `google/flan-t5-xxl`

## Ограничения Hugging Face Inference API

- **Лимит запросов**: 30,000 запросов/месяц на бесплатном tier
- **Rate limiting**: ~1,000 запросов/час
- **Латентность**: может быть выше локальных моделей (2-5 секунд)

**Решения:**
- Кеширование частых запросов
- Асинхронная обработка запросов

## Документация

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- [Langfuse Documentation](https://langfuse.com/docs)

## Docker

### Запуск через Docker Compose

```bash
# Сборка и запуск
docker-compose up --build

# В фоновом режиме
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

Streamlit UI будет доступен на `http://localhost:8501`

### Запуск только Streamlit через Docker

```bash
# Сборка образа
docker build -t electrolabbot .

# Запуск контейнера
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  electrolabbot
```

## Устранение неполадок

### Ошибка: "HUGGINGFACE_API_TOKEN не найден"

Убедитесь, что токен установлен в `.env` файле:
```bash
export HUGGINGFACE_API_TOKEN=your_token_here
```

### Ошибка: "Модель недоступна через Inference API"

Некоторые модели требуют принятия условий использования:
1. Перейдите на страницу модели на Hugging Face
2. Примите условия использования
3. Попробуйте снова

### Медленная генерация ответов

- Используйте более легкие модели (например, `mistralai/Mistral-7B-Instruct-v0.2` или `meta-llama/Llama-3.1-8B-Instruct`)
- Уменьшите `k` (количество документов для извлечения)
- Используйте локальные модели через `transformers`, или решение от провайдера api

### Ошибка при парсинге PDF/DOCX

Убедитесь, что установлены все зависимости:
```bash
pip install pypdf python-docx
```

### Проблемы с Docker

- Убедитесь, что `.env` файл существует и содержит все необходимые переменные
- Проверьте, что порты 8501 и 8000 не заняты другими приложениями
- Для просмотра логов: `docker-compose logs rag-bot`

## Лицензия

MIT

