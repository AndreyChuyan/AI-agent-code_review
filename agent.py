import os
import sys
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import tiktoken
from openai import OpenAI, AuthenticationError, OpenAIError

# ========== Константы и настройка
MODEL = "gpt-4o"
MAX_TOKENS_PER_CHUNK = 3000          # ≈ 6k слов
MAX_TOKENS_IN_COMPLETION = 1500      # оставляем запас под ответ

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# ========== Загрузка окружения
load_dotenv()

for var in ["HTTP_PROXY", "HTTPS_PROXY"]:
    value = os.getenv(var)
    if value:
        os.environ[var] = value
        logging.info(f"✅ Установлена переменная окружения {var}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("❌ Не указан OPENAI_API_KEY")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ========== Вспомогательные функции
def validate_token() -> bool:
    """Проверяет, работает ли API-ключ OpenAI."""
    try:
        client.models.list()
        logging.info("✅ API-ключ успешно проверен.")
        return True
    except AuthenticationError:
        logging.error("❌ Неверный API-ключ: OpenAI отказал в доступе.")
        return False
    except OpenAIError as e:
        logging.warning(f"⚠️ Ошибка при проверке токена: {e}")
        return False

def encoder_for(model: str):
    """
    В tiktoken gpt-4o пока может не быть, поэтому берём базовый cl100k_base.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str) -> int:
    """Подсчитывает количество токенов в тексте для заданной модели."""
    enc = encoder_for(model)
    return len(enc.encode(text))

def chunk_text(text: str, model: str, max_tokens: int) -> List[str]:
    """
    Разбиваем текст по токенам, а не по словам.
    """
    enc = encoder_for(model)
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


# ========== OpenAI вызовы
def chat_completion(messages, model=MODEL, temperature=0.2):
    """
    Единственный вход в OpenAI. Перехватываем ошибки.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS_IN_COMPLETION,
        )
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        logging.error(f"❌ OpenAI error: {e}")
        return ""


# ========== Бизнес-логика 
def summarise_file(fp: Path) -> str:
    """
    Делаем краткий конспект файла, чтобы позже
    можно было строить сводный обзор.
    """
    text = fp.read_text(encoding="utf-8")
    chunks = chunk_text(text, MODEL, MAX_TOKENS_PER_CHUNK)

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": "Ты — эксперт-ревьюер кода. Сделай очень краткое (5-7 пунктов) резюме содержания кода ниже. Не пиши улучшения, только что делает код.  Ответ на русском языке."},
            {"role": "user", "content": chunk},
        ]
        summary = chat_completion(messages)
        summaries.append(f"– Часть {i}: {summary}")
    return "\n".join(summaries)

def review_single_file(fp: Path):
    """
    Анализ единичного файла
    """
    if not fp.is_file():
        logging.warning(f"⚠️ Файл {fp} не найден — пропускаем")
        return

    text = fp.read_text(encoding="utf-8")
    chunks = chunk_text(text, MODEL, MAX_TOKENS_PER_CHUNK)

    logging.info(f"=== Код-ревью файла {fp.name} ===")
    for i, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": "Ты — эксперт-ревьюер кода. Найди баги, антипаттерны, предложи улучшения и краткие примеры, сделай итоговую оценку кода. Ответ на русском языке."},
            {"role": "user", "content": f"Часть {i} файла {fp.name}:\n{chunk}"},
        ]
        answer = chat_completion(messages)
        logging.info(f"--- Результат части {i} ---\n{answer}\n")


def review_multiple_files(files: List[Path]):
    """
    Анализ несколькьких файлов всовокупности
    """
    # 1. Получаем короткие резюме каждого файла
    file_summaries = {}
    for fp in files:
        if not fp.is_file():
            logging.warning(f"⚠️ {fp} не найден — пропускаем")
            continue
        logging.info(f"📄 Конспектируем {fp.name} ...")
        file_summaries[fp.name] = summarise_file(fp)
    # 2. Собираем единый «контекст проекта»
    project_context = []
    for name, summary in file_summaries.items():
        project_context.append(f"### {name}\n{summary}")
    project_overview = "\n\n".join(project_context)
    # 3. Делим по токенам и отправляем на архитектурный анализ
    chunks = chunk_text(project_overview, MODEL, MAX_TOKENS_PER_CHUNK)
    logging.info("=== Комплексный обзор проекта ===")
    for i, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": "Ты — эксперт архитектуры ПО. Проанализируй следующие файлы совместно, найди проблемы во взаимодействии между модулями, предложи рефакторинг при необходимости, дай краткие примеры, сделай итоговую оценку кода.. Файлы проекта: " + ", ".join(file_summaries.keys()) +". Ответ на русском."},
            {"role": "user", "content": f"Часть {i} сводного конспекта:\n{chunk}"},
        ]
        answer = chat_completion(messages)
        logging.info(f"--- Результат части {i} ---\n{answer}\n")


def review_code(paths: List[Path]):
    if len(paths) == 1:
        review_single_file(paths[0])
    else:
        review_multiple_files(paths)


# ========== Точка входа
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Использование: python agent.py путь/к/файлу1.py [файл2.js ...]")
        sys.exit(1)

    if not validate_token():
        sys.exit(1)

    targets = [Path(p) for p in sys.argv[1:]]
    review_code(targets)