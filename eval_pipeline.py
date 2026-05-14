import argparse
import os
import csv
import json
import anthropic
import openai
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

BASE_DIR = Path(__file__).parent
FERPA_PATH = BASE_DIR / "ferpa.txt"
BENCHMARK_PATH = BASE_DIR / "benchmark.jsonl"
RUNS_DIR = BASE_DIR / "items"

SYSTEM_HEADER = "You are a legal expert specializing in FERPA (the Family Educational Rights and Privacy Act)."

JUDGE_QUESTIONS = [
    "Does the LLM RESPONSE recognize how answering the user's original question depends on having the information solicited in the GOLD EXCCHANGE'S \"Expected follow-up question\"? Note: this is a test of the system's ability to understand dependencies, not of its ability to ask follow-up questions. Answer only YES or NO.",
    "Does the LLM RESPONSE provide the correct Yes/No query response as shown in the GOLD EXCHANGE? Answer only YES or NO.",
    "Does the LLM RESPONSE include all of the additional contextual information that features in the final part of the GOLD EXCHANGE? Answer only YES or NO.",
]


def get_gpt_response(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def get_claude_response(prompt: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def get_gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-3.1-pro-preview")
    response = model.generate_content(prompt)
    return response.text


def get_judge_answer(question: str, response_text: str, expected_response: str, max_tokens: int = 2048) -> str:
    judge_prompt = (
        f"GOLD EXCHANGE:\n\n{expected_response}\n\n"
        f"LLM RESPONSE:\n\n{response_text}\n\n"
        f"{question}"
    )
    result = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return next(b.text for b in result.content if b.type == "text").strip().upper()


def load_benchmark(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct-judge", action="store_true")
    args = parser.parse_args()

    ferpa_text = FERPA_PATH.read_text(encoding="latin-1")
    records = sorted(load_benchmark(BENCHMARK_PATH), key=lambda r: int(r["task_id"]))

    getters = {
        "gpt-5.4": get_gpt_response,
        "claude-opus-4-6": get_claude_response,
        "gemini-3.1-pro-preview": get_gemini_response,
    }

    if args.correct_judge:
        for record in records:
            item_num = record["task_id"]
            item_dir = RUNS_DIR / item_num
            expected_response = record["gold_exchange"]
            for model_name in getters:
                judge_file = item_dir / f"{model_name}_judge.json"
                if not judge_file.exists():
                    continue
                judge_answers = json.loads(judge_file.read_text())
                if "ERROR" not in judge_answers:
                    continue
                response_file = item_dir / f"{model_name}.txt"
                if not response_file.exists():
                    continue
                model_response = response_file.read_text()
                print(f"Item {item_num} {model_name}: re-judging errors...")
                for i, q in enumerate(JUDGE_QUESTIONS):
                    if judge_answers[i] != "ERROR":
                        continue
                    if i == 1 and judge_answers[0] == "NO":
                        judge_answers[i] = "NO"
                        continue
                    try:
                        answer = get_judge_answer(q, model_response, expected_response, max_tokens=4096)
                        if "YES" in answer:
                            answer = "YES"
                        elif "NO" in answer:
                            answer = "NO"
                    except Exception as e:
                        print(f"    Judge error: {e}")
                        answer = "ERROR"
                    judge_answers[i] = answer
                judge_file.write_text(json.dumps(judge_answers))
    else:
        for record in records:
            item_num = record["task_id"]
            item_dir = RUNS_DIR / item_num
            item_dir.mkdir(parents=True, exist_ok=True)
            user_query = record["user_query"]
            prompt = ferpa_text + "\n\n" + SYSTEM_HEADER + "\n\n" + user_query
            expected_response = record["gold_exchange"]

            print(f"Processing item {item_num}...")

            for model_name, getter in getters.items():
                judge_file = item_dir / f"{model_name}_judge.json"
                if judge_file.exists():
                    print(f"  {model_name}: skipping (already judged)")
                    continue

                response_file = item_dir / f"{model_name}.txt"
                if response_file.exists():
                    print(f"  {model_name}: loading cached response")
                    model_response = response_file.read_text()
                else:
                    print(f"  {model_name}: querying...")
                    try:
                        model_response = getter(prompt)
                        response_file.write_text(model_response)
                    except Exception as e:
                        print(f"  {model_name}: ERROR - {e}")
                        model_response = f"ERROR: {e}"
                        response_file.write_text(model_response)

                print(f"  {model_name}: judging...")
                judge_answers = []
                for i, q in enumerate(JUDGE_QUESTIONS):
                    if i == 1 and judge_answers[0] == "NO":
                        judge_answers.append("NO")
                        continue
                    try:
                        answer = get_judge_answer(q, model_response, expected_response)
                        if "YES" in answer:
                            answer = "YES"
                        elif "NO" in answer:
                            answer = "NO"
                    except Exception as e:
                        print(f"    Judge error: {e}")
                        answer = "ERROR"
                    judge_answers.append(answer)
                judge_file.write_text(json.dumps(judge_answers))

    rows = []
    for record in records:
        item_num = record["task_id"]
        item_dir = RUNS_DIR / item_num
        for model_name in getters:
            judge_file = item_dir / f"{model_name}_judge.json"
            if judge_file.exists():
                judge_answers = json.loads(judge_file.read_text())
                rows.append({
                    "item": item_num,
                    "model": model_name,
                    "q1_recognizes_followup": judge_answers[0],
                    "q2_correct_yes_no": judge_answers[1],
                    "q3_includes_context": judge_answers[2],
                })

    csv_path = BASE_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["item", "model", "q1_recognizes_followup", "q2_correct_yes_no", "q3_includes_context"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
