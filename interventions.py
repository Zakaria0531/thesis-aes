"""Generate GPT-4o-mini rewrites of ASAP-2 essays at three intervention levels (light/medium/heavy)."""

import concurrent.futures
import logging
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from modules.prompts import LEVELS, PROMPTS

CSV_PATH = Path("../asap-aes/ASAP2_train_sourcetexts.csv")
DATA_DIR = Path("data")
MODEL    = "gpt-4o-mini"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def sanitize_folder_name(name: str) -> str:
    return re.sub(r"""["'?]""", "", name).strip().replace(" ", "_")


def prompt_dir(prompt_name: str, data_dir: Path) -> Path:
    return data_dir / sanitize_folder_name(prompt_name)


def save_originals_and_scores(prompt_name: str, group: pd.DataFrame, data_dir: Path) -> None:
    orig_dir = prompt_dir(prompt_name, data_dir) / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for _, row in group.iterrows():
        (orig_dir / f"{row['essay_id']}.txt").write_text(row["full_text"], encoding="utf-8")
    group[["essay_id", "score"]].to_csv(prompt_dir(prompt_name, data_dir) / "scores.csv", index=False)
    logger.info("[%s] %d originals + scores.csv saved", prompt_name, len(group))


def process_single(
    essay_id: str,
    full_text: str,
    out_path: Path,
    system_prompt: str,
    client: OpenAI,
    model: str,
) -> str:
    """Call the API for one essay, write result to out_path, return essay_id."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=42,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": full_text},
        ],
    )
    out_path.write_text(response.choices[0].message.content.strip(), encoding="utf-8")
    return essay_id


def apply_intervention(
    prompt_name: str,
    group: pd.DataFrame,
    level: str,
    client: OpenAI,
    data_dir: Path,
    model: str = MODEL,
    max_workers: int = 10,
) -> None:
    """Apply one intervention level to all essays in a prompt group, concurrently and idempotently."""
    out_dir = prompt_dir(prompt_name, data_dir) / level
    out_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, str, Path]] = []
    skipped = 0
    for row in group.itertuples(index=False):
        out_path = out_dir / f"{row.essay_id}.txt"
        if out_path.exists():
            skipped += 1
        else:
            pending.append((row.essay_id, row.full_text, out_path))

    if skipped:
        logger.info("[%s/%s] %d/%d already done, processing %d ...",
                    prompt_name, level, skipped, len(group), len(pending))

    completed = errors = 0
    system_prompt = PROMPTS[level]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[concurrent.futures.Future, str] = {
            executor.submit(process_single, eid, text, path, system_prompt, client, model): eid
            for eid, text, path in pending
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error("[%s/%s] ERROR — %s: %s", prompt_name, level, futures[future], exc)
                errors += 1
            completed += 1
            if completed % 20 == 0 or completed == len(pending):
                logger.info("[%s/%s] %d/%d done (%d errors, %d skipped)",
                            prompt_name, level, completed, len(pending), errors, skipped)

    logger.info("[%s/%s] finished — %d written, %d skipped, %d errors",
                prompt_name, level, len(pending) - errors, skipped, errors)


def main() -> None:
    load_dotenv()
    client = OpenAI()

    df = pd.read_csv(CSV_PATH, usecols=["essay_id", "full_text", "score", "prompt_name"])
    logger.info("Loaded %d essays across %d prompts", len(df), df["prompt_name"].nunique())

    for prompt_name, group in df.groupby("prompt_name"):
        save_originals_and_scores(prompt_name, group, DATA_DIR)

    for level in LEVELS:
        logger.info("Starting '%s' intervention ...", level)
        for prompt_name, group in df.groupby("prompt_name"):
            apply_intervention(prompt_name, group, level, client, DATA_DIR)

    logger.info("All interventions complete.")


if __name__ == "__main__":
    main()