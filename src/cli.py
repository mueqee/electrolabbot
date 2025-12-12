"""
CLI-интерфейс для RAG-бота.
Позволяет задавать вопросы через командную строку.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

from rag_service import create_rag_service, RAGService

# Загружаем переменные окружения
load_dotenv()

console = Console()


def print_answer(question: str, result: dict, service: RAGService):
    """Красиво выводит ответ на вопрос."""
    # Заголовок с вопросом
    console.print(Panel(f"[bold cyan]{question}[/bold cyan]", title="Вопрос", border_style="cyan"))
    
    # Ответ
    console.print("\n[bold green]Ответ:[/bold green]")
    console.print(Markdown(result["answer"]))
    
    # Источники
    if result["sources"]:
        console.print(f"\n[bold yellow]Источники ({len(result['sources'])}):[/bold yellow]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim", width=3)
        table.add_column("Документ", style="cyan")
        table.add_column("Ревизия", style="yellow")
        table.add_column("Тип", style="green")
        table.add_column("Фрагмент", style="dim", max_width=50)
        
        for i, source in enumerate(result["sources"], 1):
            table.add_row(
                str(i),
                source["name"],
                source["revision"],
                source["type"],
                source["snippet"][:100] + "..." if len(source["snippet"]) > 100 else source["snippet"]
            )
        
        console.print(table)
    
    # Trace ID
    if "trace_id" in result:
        console.print(f"\n[dim]Langfuse Trace ID: {result['trace_id']}[/dim]")


def interactive_mode(service: RAGService):
    """Интерактивный режим для задавания вопросов."""
    console.print("[bold green]Интерактивный режим RAG-бота[/bold green]")
    console.print("[dim]Введите 'exit' или 'quit' для выхода[/dim]\n")
    
    while True:
        try:
            question = console.input("[bold cyan]Ваш вопрос: [/bold cyan]")
            
            if question.lower() in ["exit", "quit", "выход"]:
                console.print("[yellow]До свидания![/yellow]")
                break
            
            if not question.strip():
                continue
            
            console.print("[dim]Обработка...[/dim]")
            
            result = service.ask(question)
            print_answer(question, result, service)
            console.print("\n" + "="*80 + "\n")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Прервано пользователем[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Ошибка:[/bold red] {e}")


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="CLI для инженерного RAG-бота на базе Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python cli.py "Какой класс бетона используется для фундаментов?"
  python cli.py --interactive
  python cli.py "Вопрос" --model "mistralai/Mistral-7B-Instruct-v0.2"
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Вопрос для задавания боту"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Запустить в интерактивном режиме"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Название модели Hugging Face (по умолчанию из переменных окружения)"
    )
    
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Количество документов для извлечения (по умолчанию: 3)"
    )
    
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Отключить Langfuse трейсинг"
    )
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(1)
    
    # Инициализация сервиса
    try:
        console.print("[dim]Инициализация RAG-сервиса...[/dim]")
        service = create_rag_service(
            model_name=args.model,
            k=args.k,
            enable_langfuse=not args.no_langfuse
        )
        console.print("[green]✅ RAG-сервис готов[/green]\n")
    except Exception as e:
        console.print(f"[bold red]Ошибка при инициализации:[/bold red] {e}")
        console.print("\n[dim]Убедитесь, что:")
        console.print("  1. HUGGINGFACE_API_TOKEN установлен")
        console.print("  2. Выполнен ingest.py для создания векторной базы")
        console.print("  3. Установлены все зависимости[/dim]")
        sys.exit(1)
    
    # Интерактивный режим
    if args.interactive:
        interactive_mode(service)
    else:
        # Одиночный вопрос
        try:
            console.print("[dim]Обработка...[/dim]")
            result = service.ask(args.question)
            print_answer(args.question, result, service)
        except Exception as e:
            console.print(f"[bold red]Ошибка:[/bold red] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

