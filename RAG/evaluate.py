"""
Оценка качества RAG-системы.

Метрики:
- Faithfulness
- Answer Relevance
- Refusal Accuracy
- Correction Rate

Требования:
    pip install pandas openpyxl python-dotenv langchain langchain-openai
    OPENROUTER_API_KEY должен быть в .env
"""

import os
import re
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Конфигурация 
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def get_llm() -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("Ключ не найден! Создайте файл .env с OPENROUTER_API_KEY=sk-or-...")
    return ChatOpenAI(
        model=MODEL,
        temperature=0,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )


# Промпты для оценки

FAITHFULNESS_PROMPT = """Ты — строгий эксперт-оценщик RAG-систем. Оцени, насколько ответ основан только на документе.

Вопрос: {question}
Ответ системы: {answer}

Критерии:
- 1.0: Ответ содержит только информацию из документа или явный отказ при отсутствии данных
- 0.5: Ответ в основном корректен, но содержит незначительные домыслы
- 0.0: Ответ содержит выдуманные факты или не связан с документом

Ответь ТОЛЬКО числом: 1.0, 0.5 или 0.0"""

RELEVANCE_PROMPT = """Ты — строгий эксперт-оценщик. Оцени, насколько ответ отвечает на вопрос.

Вопрос: {question}
Ответ системы: {answer}

Критерии:
- 1.0: Ответ полностью и точно отвечает на вопрос
- 0.5: Ответ частично отвечает на вопрос или неполный
- 0.0: Ответ не отвечает на вопрос

Ответь ТОЛЬКО числом: 1.0, 0.5 или 0.0"""


def extract_score(text: str) -> float:
    """Извлекаем числовую оценку из ответа LLM."""
    match = re.search(r"\b(1\.0|0\.5|0\.0)\b", text)
    if match:
        return float(match.group(1))
    if "1" in text:
        return 1.0
    if "0.5" in text or "0,5" in text:
        return 0.5
    return 0.0


def score_faithfulness(llm: ChatOpenAI, question: str, answer: str) -> float:
    prompt = FAITHFULNESS_PROMPT.format(question=question, answer=answer)
    response = llm.invoke(prompt)
    return extract_score(response.content)


def score_relevance(llm: ChatOpenAI, question: str, answer: str) -> float:
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    response = llm.invoke(prompt)
    return extract_score(response.content)


def is_refusal(answer: str) -> bool:
    """Проверяем, что ответ содержит явный отказ (для вопросов-ловушек)."""
    keywords = [
        "отсутствует в документе",
        "нет в документе",
        "не содержится",
        "отказываюсь",
        "не могу выдумывать",
        "отвечаю строго на основе",
        "данная информация отсутствует",
    ]
    answer_lower = answer.lower()
    return any(kw in answer_lower for kw in keywords)


def is_correction(answer: str) -> bool:
    """Проверяем, что ответ исправляет ошибку из вопроса."""
    keywords = [
        "ошибку",
        "ошибка",
        "не совпадает",
        "неверн",
        "содержит ошибку",
        "правильный ответ",
        "согласно стратегии",
    ]
    answer_lower = answer.lower()
    return any(kw in answer_lower for kw in keywords)


# Типизация вопросов

def classify_question(question: str, answer: str) -> str:
    q_lower = question.lower()
    a_lower = answer.lower()

    # Prompt injection
    if any(w in q_lower for w in ["игнорируй", "придумай", "притворись", "забудь"]):
        return "prompt_injection"

    # Вопрос о несуществующих данных
    if any(w in a_lower for w in ["отсутствует в документе", "нет в документе"]):
        return "absent_data"

    # Фактическая ошибка в вопросе (неверная цифра или утверждение)
    # Маркеры в самом вопросе:
    error_markers_in_question = [
        "12 млрд",      # конкретная неверная цифра из вашего теста
        "неверн",
        "ошибк",
        "должно быть",
    ]
    if any(m in q_lower for m in error_markers_in_question):
        return "factual_error"

    # Маркеры исправления в ответе:
    correction_markers = ["ошибк", "содержит ошибку", "неверн", "исправл", "должно быть"]
    if any(m in a_lower for m in correction_markers):
        return "factual_error"

    return "factual"

# Основная функция оценки

def evaluate(test_set_path: str, output_path: str = "evaluation_results.json") -> dict:
    df = pd.read_excel(test_set_path)

    print(f"Файл загружен. Столбцы: {df.columns.tolist()}")
    print(f"   Строк всего: {len(df)}")

    df.columns = df.columns.str.strip().str.lower()

    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(
            f"Не найдены столбцы 'question' и 'answer'.\n"
            f"Фактические столбцы: {df.columns.tolist()}\n"
            f"Переименуйте столбцы в файле или передайте нужные имена."
        )

    df = df.dropna(subset=["question", "answer"])
    print(f"Загружено пар Q&A (после фильтрации пустых): {len(df)}\n")

    if len(df) == 0:
        print("После фильтрации пустых строк данных не осталось.")
        return {}

    llm = get_llm()
    results = []

    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        q_type = classify_question(question, answer)

        print(f"[{i+1}/{len(df)}] Тип: {q_type}")
        print(f"  Q: {question[:80]}...")

        faithfulness = score_faithfulness(llm, question, answer)
        relevance = score_relevance(llm, question, answer)
        refusal_ok = is_refusal(answer) if q_type in ("prompt_injection", "absent_data") else None
        correction_ok = is_correction(answer) if q_type == "factual_error" else None

        result = {
            "question": question,
            "answer": answer,
            "type": q_type,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "refusal_correct": refusal_ok,
            "correction_correct": correction_ok,
        }
        results.append(result)

        print(f"  Faithfulness: {faithfulness} | Relevance: {relevance}", end="")
        if refusal_ok is not None:
            print(f" | Refusal: {'ok' if refusal_ok else 'not ok'}", end="")
        if correction_ok is not None:
            print(f" | Correction: {'ok' if correction_ok else 'not ok'}", end="")
        print()

    # Агрегированные метрики
    total = len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / total
    avg_relevance = sum(r["relevance"] for r in results) / total

    refusal_cases = [r for r in results if r["refusal_correct"] is not None]
    refusal_accuracy = (
        sum(1 for r in refusal_cases if r["refusal_correct"]) / len(refusal_cases)
        if refusal_cases else None
    )

    correction_cases = [r for r in results if r["correction_correct"] is not None]
    correction_rate = (
        sum(1 for r in correction_cases if r["correction_correct"]) / len(correction_cases)
        if correction_cases else None
    )

    summary = {
        "total_questions": total,
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_relevance": round(avg_relevance, 3),
        "refusal_accuracy": round(refusal_accuracy, 3) if refusal_accuracy is not None else "N/A",
        "correction_rate": round(correction_rate, 3) if correction_rate is not None else "N/A",
        "details": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Вывод итогов 
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)
    print(f"  Faithfulness (достоверность)  : {avg_faithfulness:.3f}")
    print(f"  Answer Relevance (релевантность): {avg_relevance:.3f}")
    if refusal_accuracy is not None:
        print(f"  Refusal Accuracy (отказы)      : {refusal_accuracy:.3f}  ({len(refusal_cases)} вопросов)")
    if correction_rate is not None:
        print(f"  Correction Rate (исправления)  : {correction_rate:.3f}  ({len(correction_cases)} вопросов)")
    print(f"\nДетальные результаты сохранены: {output_path}")

    return summary


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка качества RAG-системы")
    parser.add_argument(
        "--test_set",
        default="data/test_set_result.xlsx",
        help="Путь к Excel-файлу с вопросами и ответами",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Путь для сохранения JSON-отчёта",
    )
    args = parser.parse_args()
    evaluate(args.test_set, args.output)