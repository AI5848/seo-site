import os
import re
import json
import time
from datetime import datetime, timezone

from huggingface_hub import InferenceClient  # HF Inference API wrapper :contentReference[oaicite:2]{index=2}

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"  # popular open instruct model :contentReference[oaicite:3]{index=3}

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:80].strip("-") or "post"

def read_topics(path="topics.txt"):
    with open(path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return topics

def list_existing_slugs(posts_dir="_posts"):
    if not os.path.isdir(posts_dir):
        return set()
    slugs = set()
    for fn in os.listdir(posts_dir):
        # Jekyll post filename: YYYY-MM-DD-slug.md
        m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)\.md$", fn)
        if m:
            slugs.add(m.group(1))
    return slugs

def build_prompt(topic: str) -> str:
    # Ask model to output strict JSON to make parsing reliable
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

    # Serverless inference can return 503 while the model spins up :contentReference[oaicite:4]{index=4}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # text_generation works for text-generation pipeline models
            out = client.text_generation(
                prompt=prompt,
                max_new_tokens=1200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
            )
            return out
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(min(10 * attempt, 45))
    raise RuntimeError(f"HF inference failed after retries: {last_err}")

def extract_json(text: str) -> dict:
    # Try direct parse
    text = text.strip()

    # Some models wrap in ```json ... ```
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # If extra text leaked, try to find first {...} block
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

    # Pick the first topic that doesn't already exist as a slug
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
        raise ValueError("Model did not return 5 keywords.")

    # Jekyll post
    os.makedirs("_posts", exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"_posts/{today}-{chosen_slug}.md"

    tags = [slugify(k).replace("-", "_") for k in keywords]

    front_matter = f"""---
layout: post
title: "{title.replace('"', "'")}"
description: "{meta.replace('"', "'")}"
tags: [{", ".join(tags)}]
keywords: {json.dumps(keywords)}
---

"""
    # Add a small “keywords” line in body for transparency (optional)
    body = f"> **SEO Keywords:** {', '.join(keywords)}\n\n{article}\n"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(front_matter + body)

    print(f"Created: {filename}")

if __name__ == "__main__":
    main()
