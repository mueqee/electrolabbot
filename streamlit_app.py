"""
Streamlit UI для RAG-бота.
Предоставляет веб-интерфейс для взаимодействия с RAG-системой.
"""

import os
import sys
import streamlit as st
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_service import create_rag_service, RAGService
from dotenv import load_dotenv

load_dotenv()

# настройка страницы
st.set_page_config(
    page_title="ЭлектроЛаббот - RAG-ассистент по электротехническим отчётам",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# инициализация session state
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def initialize_service() -> Optional[RAGService]:
    """Инициализирует RAG-сервис."""
    try:
        model_name = os.getenv("HF_LLM_MODEL")
        enable_langfuse = os.getenv("ENABLE_LANGFUSE", "true").lower() == "true"
        enable_memory = True  # всегда включена memory для UI
        
        service = create_rag_service(
            model_name=model_name,
            enable_langfuse=enable_langfuse,
            enable_memory=enable_memory
        )
        return service
    except Exception as e:
        st.error(f"Ошибка при инициализации сервиса: {e}")
        st.info("Убедитесь, что:\n"
                "1. HUGGINGFACE_API_TOKEN установлен в .env\n"
                "2. Выполнен ingest.py для создания векторной базы\n"
                "3. Установлены все зависимости")
        return None


def main():
    """Главная функция Streamlit приложения."""
    
    # заголовок
    st.title("⚡")
    st.markdown("**RAG-ассистент по приёмо-сдаточным и эксплуатационным испытаниям электроустановок**")
    st.markdown("---")
    
    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # кнопка инициализации
        if st.button("🔄 Инициализировать сервис", type="primary"):
            with st.spinner("Инициализация RAG-сервиса..."):
                st.session_state.rag_service = initialize_service()
                if st.session_state.rag_service:
                    st.session_state.initialized = True
                    st.success("✅ Сервис инициализирован!")
                    # Очищаем историю при переинициализации
                    st.session_state.messages = []
                else:
                    st.session_state.initialized = False
        
        st.divider()
        
        # информация о сервисе
        if st.session_state.initialized and st.session_state.rag_service:
            st.success("✅ Сервис активен")
            st.info(f"**Модель:** {st.session_state.rag_service.model_name}")
            st.info(f"**Memory:** {'Включена' if st.session_state.rag_service.enable_memory else 'Выключена'}")
            st.info(f"**Langfuse:** {'Включен' if st.session_state.rag_service.enable_langfuse else 'Выключен'}")
            
            # кнопка очистки истории
            if st.button("🗑️ Очистить историю диалога"):
                if st.session_state.rag_service:
                    st.session_state.rag_service.clear_memory()
                st.session_state.messages = []
                st.success("История очищена!")
                st.rerun()
        else:
            st.warning("⚠️ Сервис не инициализирован")
        
        st.divider()
        
        # Информация
        st.markdown("### 📖 Информация")
        st.markdown("""
        **База знаний:**
        - ПУЭ, ПТЭЭП, СП, ГОСТ, РД
        - 15 анонимизированных технических отчётов
        - Справочники по оборудованию
        
        **Возможности:**
        - Поиск нормативных требований
        - Интерпретация результатов измерений
        - Примеры из реальных отчётов
        - Помощь в заполнении протоколов
        
        **Использование:**
        1. Нажмите "Инициализировать сервис"
        2. Задайте вопрос в поле ввода
        3. Получите ответ с источниками
        """)
        
        st.markdown("### 💡 Примеры вопросов")
        example_questions = [
            "Какие требования к сопротивлению заземления по ПУЭ?",
            "Сопротивление изоляции 210 МОм - это нормально?",
            "Покажи пример протокола проверки автоматических выключателей",
            "Какие поля должны быть в протоколе согласования параметров цепи фаза-нуль?",
            "Какие нормы по проверке автоматических выключателей в ПУЭ?",
            "Покажи пример отчёта по стадиону",
            "Какое допустимое сопротивление изоляции для кабеля 5x70?",
            "Как интерпретировать результаты проверки цепи фаза-нуль?"
        ]
        for i, q in enumerate(example_questions, 1):
            st.caption(f"{i}. {q}")
    
    # Основная область
    if not st.session_state.initialized:
        st.info("👈 Нажмите 'Инициализировать сервис' в боковой панели, чтобы начать")
        return
    
    # Отображение истории сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Показываем источники, если есть
            if "sources" in message and message["sources"]:
                with st.expander("📚 Источники"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source['name']}** (ревизия: {source['revision']})")
                        st.caption(f"Тип: {source['type']} | Категория: {source['category']}")
                        st.text(source['snippet'][:200] + "..." if len(source['snippet']) > 200 else source['snippet'])
    
    # Поле ввода вопроса
    if prompt := st.chat_input("Задайте вопрос о нормативных требованиях, результатах измерений или протоколах..."):
        # Добавляем вопрос пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Генерируем ответ
        with st.chat_message("assistant"):
            with st.spinner("Поиск информации и генерация ответа..."):
                try:
                    result = st.session_state.rag_service.ask(prompt)
                    
                    # Отображаем ответ
                    st.markdown(result["answer"])
                    
                    # Отображаем источники
                    if result["sources"]:
                        with st.expander("📚 Источники"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**{i}. {source['name']}** (ревизия: {source['revision']})")
                                st.caption(f"Тип: {source['type']} | Категория: {source['category']}")
                                st.text(source['snippet'][:200] + "..." if len(source['snippet']) > 200 else source['snippet'])
                    
                    # Добавляем ответ в историю
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    # Показываем trace_id, если есть
                    if "trace_id" in result:
                        st.caption(f"🔍 Trace ID: {result['trace_id']}")
                
                except Exception as e:
                    error_msg = f"Ошибка при обработке запроса: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()

