import os
import re
import json
import time
from datetime import datetime, timezone

from huggingface_hub import InferenceClient


# Chat/conversational uyumlu bir model seçiyoruz
MODEL_ID = "google/gemma-2-2b-it"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:80].strip("-") or "post"


def read_topics(path: str = "topics.txt") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return topics


def list_existing_slugs(posts_dir: str = "_posts") -> set[str]:
    if not os.path.isdir(posts_dir):
        return set()
    slugs = set()
    for fn in os.listdir(posts_dir):
        m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)\.md$", fn)
        if m:
            slugs.add(m.group(1))
    return slugs


def build_prompt(topic: str) -> str:
    # Modelin sadece JSON döndürmesini istiyoruz (parsing için)
    return f"""
You are an SEO content writer.

Write ONE English blog post about: "{topic}".

Requirements:
- 700 to 1000 words (must be 500+).
- Use clear structure with H2/H3 headings.
- Include a short meta description (<= 160 characters).
- Provide 5 SEO keywords (as short phrases).
- Provide a compelling title (max ~70 characters).
- Natural keyword usage; do not spam.
- Add a brief FAQ section with 3 Q&As at the end.

Return ONLY valid JSON with this exact schema:
{{
  "title": "...",
  "meta_description": "...",
  "keywords": ["...","...","...","...","..."],
  "article_markdown": "Markdown content with headings"
}}
""".strip()


def call_hf(prompt: str, max_retries: int = 6) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it in GitHub Secrets as HF_TOKEN.")

    client = InferenceClient(model=MODEL_ID, token=token)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # Chat API (conversational modeller için)
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful SEO content writer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1400,
                temperature=0.7,
                top_p=0.9,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(min(10 * attempt, 45))

    raise RuntimeError(f"HF inference failed after retries: {last_err}")


def extract_json(text: str) -> dict:
    text = text.strip()

    # Model ```json ... ``` ile dönerse temizle
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Eğer başında/sonunda fazlalık varsa ilk JSON bloğunu yakala
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0)

    return json.loads(text)


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
    raw = call_hf(prompt)

    data = extract_json(raw)
    title = data["title"].strip()
    meta = data["meta_description"].strip()
    keywords = data["keywords"]
    article = data["article_markdown"].strip()

    if not isinstance(keywords, list) or len(keywords) != 5:
        raise ValueError("Model did not return exactly 5 keywords.")

    # Jekyll post dosyasını oluştur
    os.makedirs("_posts", exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"_posts/{today}-{chosen_slug}.md"

    # tag'leri basit hale getir
    tags = [slugify(k).replace("-", "_") for k in keywords]

    front_matter = f"""---
layout: post
title: "{title.replace('"', "'")}"
description: "{meta.replace('"', "'")}"
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
