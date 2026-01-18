import os
import re
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from huggingface_hub import InferenceClient

MODEL_ID = "google/gemma-2-2b-it"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:80].strip("-") or "post"


def read_topics(path: str = "topics.txt") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def list_existing_slugs(posts_dir: str = "_posts") -> set:
    if not os.path.isdir(posts_dir):
        return set()
    slugs = set()
    for fn in os.listdir(posts_dir):
        m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)\.md$", fn)
        if m:
            slugs.add(m.group(1))
    return slugs


def build_prompt(topic: str) -> str:
    # NOT: article_markdown'ı JSON içine koymak LLM'lerde sık bozuluyor.
    # Bu yüzden JSON'da sadece title/meta/keywords alıyoruz.
    # Makaleyi JSON DIŞINDA, ayrı bir blok olarak döndürmesini istiyoruz.
    return f"""
You are an SEO content writer.

Write ONE English blog post about: "{topic}".

CONTENT RULES:
- 700 to 1000 words (must be 500+).
- Use clear structure with H2/H3 headings.
- Add a brief FAQ section with 3 Q&As at the end.

OUTPUT FORMAT (VERY IMPORTANT):
1) First, output a SINGLE LINE of valid JSON (no code fences) with EXACT keys:
{{
  "title": "...",
  "meta_description": "...",
  "keywords": ["...","...","...","...","..."]
}}
- meta_description must be <= 160 characters.
- keywords must be exactly 5 short phrases.

2) After that JSON line, output:
---ARTICLE---
(then the full article in Markdown)
---END---

Do not add anything else.
""".strip()


def call_hf(prompt: str, max_retries: int = 6) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it in GitHub Secrets as HF_TOKEN.")

    client = InferenceClient(model=MODEL_ID, token=token)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful SEO content writer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1800,
                temperature=0.7,
                top_p=0.9,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(min(10 * attempt, 45))

    raise RuntimeError(f"HF inference failed after retries: {last_err}")


def parse_output(raw: str) -> Dict[str, Any]:
    """
    Beklenen format:
    <json tek satır>
    ---ARTICLE---
    <markdown>
    ---END---
    """
    raw = raw.strip()

    # code fence temizliği (model bazen ekliyor)
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # İlk satır JSON olsun diye tasarladık; yine de güvenli yakalayalım
    lines = raw.splitlines()
    if not lines:
        raise ValueError("Empty model output")

    json_line = None
    for i, line in enumerate(lines[:10]):  # ilk 10 satır içinde JSON arayalım
        line_stripped = line.strip()
        if line_stripped.startswith("{") and line_stripped.endswith("}"):
            json_line = line_stripped
            json_index = i
            break

    if json_line is None:
        # JSON bloğu arayalım
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise ValueError("Could not find JSON in model output")
        json_line = m.group(0)

    # JSON parse (tek satır olduğu için kontrol karakter riski düşük)
    meta = json.loads(json_line)

    # Article bölümü
    m2 = re.search(r"---ARTICLE---\s*(.*?)\s*---END---", raw, flags=re.S)
    if not m2:
        # olmazsa JSON'u çıkarıp kalan her şeyi article say
        # JSON'un çıktığı satırın altını alalım
        after = "\n".join(lines[json_index + 1:]) if "json_index" in locals() else raw.replace(json_line, "")
        article = after.strip()
    else:
        article = m2.group(1).strip()

    meta["article_markdown"] = article
    return meta


def validate(meta: Dict[str, Any]) -> None:
    if not isinstance(meta.get("title"), str) or not meta["title"].strip():
        raise ValueError("Missing title")
    if not isinstance(meta.get("meta_description"), str) or not meta["meta_description"].strip():
        raise ValueError("Missing meta_description")
    if len(meta["meta_description"]) > 160:
        meta["meta_description"] = meta["meta_description"][:160].rstrip()
    kw = meta.get("keywords")
    if not isinstance(kw, list) or len(kw) != 5:
        raise ValueError("keywords must be a list of exactly 5 items")
    if not isinstance(meta.get("article_markdown"), str) or len(meta["article_markdown"].split()) < 500:
        raise ValueError("article_markdown seems too short (<500 words)")


def main():
    topics = read_topics("topics.txt")
    if not topics:
        print("No topics in topics.txt")
        return

    existing = list_existing_slugs("_posts")

    chosen_topic = None
    chosen_slug = None
    for t in topics:
        s = slugify(t)
        if s not in existing:
            chosen_topic = t
            chosen_slug = s
            break

    if not chosen_topic:
        print("All topics already used. Add more topics to topics.txt")
        return

    prompt = build_prompt(chosen_topic)

    # 2 deneme: ilk çıktı bozuk gelirse aynı promptu tekrar iste
    last_err = None
    raw = None
    for attempt in range(2):
        try:
            raw = call_hf(prompt)
            meta = parse_output(raw)
            validate(meta)
            break
        except Exception as e:
            last_err = e
            time.sleep(3)
            meta = None

    if meta is None:
        # debug kolay olsun diye ham çıktıyı yazdır
        print("MODEL RAW OUTPUT (truncated):")
        print((raw or "")[:2000])
        raise RuntimeError(f"Failed to generate/parse content: {last_err}")

    title = meta["title"].strip()
    meta_desc = meta["meta_description"].strip()
    keywords = meta["keywords"]
    article = meta["article_markdown"].strip()

    os.makedirs("_posts", exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"_posts/{today}-{chosen_slug}.md"

    tags = [slugify(k).replace("-", "_") for k in keywords]

    front_matter = f"""---
layout: post
title: "{title.replace('"', "'")}"
description: "{meta_desc.replace('"', "'")}"
tags: [{", ".join(tags)}]
keywords: {json.dumps(keywords)}
---

"""
    body = f"> **SEO Keywords:** {', '.join(keywords)}\n\n{article}\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(front_matter + body)

    print(f"Created: {filename}")


if __name__ == "__main__":
    main()
