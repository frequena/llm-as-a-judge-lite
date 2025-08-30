import json
import os
import time
from datetime import datetime
from typing import List
import logging
import asyncio
import csv
import sys
import yaml

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

import dotenv
dotenv.load_dotenv()

# Headless plotting for environments without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Question(BaseModel):
    id: int
    question: str


class JudgeOutput(BaseModel):
    score: int = Field(ge=0, le=10)
    reasoning: str


class Judgment(BaseModel):
    question_id: int
    player_model: str
    judge_model: str
    score: int
    reasoning: str
    elapsed_ms: int


class PlayerAverage(BaseModel):
    model: str
    avg_score: float


class RunResult(BaseModel):
    judge_model: str
    players: List[str]
    timestamp: str
    player_averages: List[PlayerAverage]
    judgments: List[Judgment]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_output_text(resp) -> str:
    """Extract plain text from a Responses API response with robust fallbacks."""
    # Preferred helper provided by SDK
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text
    # Fallback: attempt to concatenate any text content present
    try:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                ctype = getattr(content, "type", None)
                if ctype in ("output_text", "input_text", "text"):
                    value = getattr(content, "text", None)
                    if isinstance(value, str):
                        parts.append(value)
        if parts:
            return "".join(parts)
    except Exception:
        pass
    # Last resort: string representation
    return str(resp)


def load_questions(path: str) -> List[Question]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Question(**item) for item in data]


async def main() -> None:
    # Configure logging (defaults to INFO). Logs go to stderr and won't affect final JSON on stdout.
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    # User-friendly single-line messages
    logging.basicConfig(level=log_level, format="%(message)s")
    # Silence verbose HTTP client logs (OpenAI/httpx/httpcore)
    for noisy_logger in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    use_emoji = os.getenv("LOG_EMOJI", "1") not in ("0", "false", "False")
    START_ICON = "▶️" if use_emoji else ">>"
    DONE_ICON = "✅" if use_emoji else "OK"
    DOC_ICON = "📝" if use_emoji else "..."
    WARN_ICON = "⚠️" if use_emoji else "!!"
    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "8"))

    dataset_path = os.getenv("DATASET_PATH", "data/questions.json")
    answer_prompt_path = os.getenv("PLAYER_PROMPT_PATH", "prompts/player.txt")
    judge_prompt_path = os.getenv("JUDGE_PROMPT_PATH", "prompts/judge.txt")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set")
        return

    # Load non-secret config (judge and players) from YAML with env overrides
    config_path = os.getenv("CONFIG_PATH", "config/bench.yaml")
    yaml_config = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
    except Exception:
        yaml_config = {}

    judge_model = os.getenv("JUDGE_MODEL") or yaml_config.get("judge_model") or "gpt-5"
    players_value = os.getenv("PLAYERS")
    if players_value:
        players: List[str] = [p.strip() for p in players_value.split(",") if p.strip()]
    else:
        players_list = yaml_config.get("players") or [
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1",
            "gpt-4-mini",
        ]
        players = [str(p).strip() for p in players_list if str(p).strip()]

    client = AsyncOpenAI(api_key=api_key)

    questions = load_questions(dataset_path)
    answer_system = load_text(answer_prompt_path)
    judge_system = load_text(judge_prompt_path)

    logging.info(
        ("🚀 " if use_emoji else "RUN ")
        + "Judge run • model: %s • players: %s • questions: %d • dataset: %s",
        judge_model,
        ", ".join(players),
        len(questions),
        dataset_path,
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def eval_pass(q: Question, player: str) -> Judgment:
        logging.info("%s Q#%s • %s", START_ICON, q.id, player)
        start = time.monotonic()

        async with semaphore:
            answer_resp = await client.responses.create(
                model=player,
                input=[
                    {"role": "system", "content": answer_system},
                    {"role": "user", "content": q.question},
                ],
            )
        answer_text = get_output_text(answer_resp).strip()
        logging.debug("%s Answer ready • %d chars", DOC_ICON, len(answer_text))

        judge_user = (
            "You will grade an answer to a question. Return strict JSON only.\n\n"
            f"Question: {q.question}\n\n"
            f"Answer (from {player}): {answer_text}\n\n"
            "Return JSON with keys 'score' (0-10) and 'reasoning' (string)."
        )

        async with semaphore:
            judge_resp = await client.responses.create(
                model=judge_model,
                input=[
                    {"role": "system", "content": judge_system},
                    {"role": "user", "content": judge_user},
                ],
            )
        judge_text = get_output_text(judge_resp).strip()

        try:
            parsed = JudgeOutput(**json.loads(judge_text))
        except Exception:
            logging.warning("%s Judge output not JSON • using fallback score", WARN_ICON)
            fallback_score = 0
            for tok in judge_text.split():
                if tok.isdigit():
                    n = int(tok)
                    if 0 <= n <= 10:
                        fallback_score = n
                        break
            parsed = JudgeOutput(score=fallback_score, reasoning=judge_text[:200])

        elapsed_ms = int((time.monotonic() - start) * 1000)
        j = Judgment(
            question_id=q.id,
            player_model=player,
            judge_model=judge_model,
            score=parsed.score,
            reasoning=parsed.reasoning,
            elapsed_ms=elapsed_ms,
        )
        logging.info("%s Q#%s • %s • score %d/3 • %d ms", DONE_ICON, q.id, player, parsed.score, elapsed_ms)
        return j

    tasks = [eval_pass(q, player) for q in questions for player in players]
    judgments: List[Judgment] = await asyncio.gather(*tasks)

    player_averages: List[PlayerAverage] = []
    for player in players:
        scores = [j.score for j in judgments if j.player_model == player]
        avg = sum(scores) / max(1, len(scores))
        player_averages.append(PlayerAverage(model=player, avg_score=round(avg, 2)))

    # Friendly summary to stderr before printing CSV to stdout
    if player_averages:
        pretty_avgs = ", ".join(f"{pa.model} {pa.avg_score}" for pa in player_averages)
        logging.info(("📊 " if use_emoji else "AVG ") + "Averages • %s", pretty_avgs)

    # CSV output to file (default results_{timestamp}.csv) and optional stdout
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path_env = os.getenv("CSV_PATH")
    if csv_path_env:
        csv_path = csv_path_env
    else:
        csv_path = f"results_{ts}.csv"
    print_csv = os.getenv("PRINT_CSV", "1") not in ("0", "false", "False")
    headers = ["question_id", "player_model", "judge_model", "score", "elapsed_ms", "reasoning"]
    player_index = {p: i for i, p in enumerate(players)}
    ordered = sorted(
        judgments,
        key=lambda x: (x.question_id, player_index.get(x.player_model, 1_000_000)),
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for j in ordered:
            writer.writerow([j.question_id, j.player_model, j.judge_model, j.score, j.elapsed_ms, j.reasoning])
    logging.info(("💾 " if use_emoji else "CSV ") + "Saved table to %s", csv_path)

    if print_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        for j in ordered:
            writer.writerow([j.question_id, j.player_model, j.judge_model, j.score, j.elapsed_ms, j.reasoning])

    # 100% stacked bar chart of score proportions per player (0/1/2/3)
    try:
        # Build per-player distributions
        player_to_scores: dict[str, List[int]] = {p: [] for p in players}
        for j in judgments:
            if j.player_model in player_to_scores:
                # Clamp scores into 0..3 for consistent comparison
                s = j.score
                if s < 0:
                    s = 0
                if s > 3:
                    s = 3
                player_to_scores[j.player_model].append(int(s))

        labels = list(players)
        n = len(labels)
        buckets = [0, 1, 2, 3]

        # Compute percentages per bucket for each player
        perc_by_bucket = {b: [] for b in buckets}
        for p in labels:
            scores = player_to_scores.get(p, [])
            total = len(scores)
            if total == 0:
                for b in buckets:
                    perc_by_bucket[b].append(0.0)
            else:
                counts = {b: 0 for b in buckets}
                for s in scores:
                    counts[s] = counts.get(s, 0) + 1
                for b in buckets:
                    perc_by_bucket[b].append(100.0 * counts[b] / total)

        # Order players on x-axis by proportion of score 3 (descending)
        order_indices = sorted(range(n), key=lambda i: perc_by_bucket[3][i], reverse=True)
        labels = [labels[i] for i in order_indices]
        for b in buckets:
            perc_by_bucket[b] = [perc_by_bucket[b][i] for i in order_indices]

        # Plot stacked bars
        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
        x = list(range(n))
        bottoms = [0.0] * n
        colors = {
            0: "#d73027",
            1: "#fc8d59",
            2: "#91bfdb",
            3: "#4575b4",
        }

        for b in buckets:
            heights = perc_by_bucket[b]
            ax.bar(x, heights, bottom=bottoms, color=colors.get(b, None), edgecolor="black", label=str(b), alpha=0.9)
            bottoms = [bot + h for bot, h in zip(bottoms, heights)]

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("percentage of judgments (%)")
        ax.set_title("Score Composition by Model (100% Stacked, Scores 0-3)")
        ax.legend(title="score", loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()

        stacked_path = os.getenv("STACKED_PLOT_PATH", f"plot_{ts}.png")
        fig.savefig(stacked_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logging.info(("🖼️ " if use_emoji else "PLOT ") + "Saved 100%% stacked plot to %s", stacked_path)
    except Exception as e:
        logging.warning(("⚠️ " if use_emoji else "!! ") + "Could not create stacked plot: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
