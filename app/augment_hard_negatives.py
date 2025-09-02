# augment_hard_negatives.py
import json, random, re, hashlib
from pathlib import Path

random.seed(42)

SRC = "converted_dataset_clean.json"          # your original file
DST = "converted_dataset_hardneg.json"        # new augmented file

TECH_POOL = [
    "python","java","spring","spring boot","react","next.js","node.js","express",
    "docker","kubernetes","terraform","ansible","mysql","postgresql","mongodb","redis",
    "elasticsearch","spark","airflow","hadoop","kafka","selenium","cypress",
    "aws","azure","gcp","lambda","s3","cloud run","cloud functions","api gateway",
    "flask","django","fastapi","rails","laravel","flutter","react native","vue","angular",
    "git","jira","confluence","jenkins","github actions","circleci","sonarqube"
]

# --- helpers ----------------------------------------------------------------
Q_RE = re.compile(r"question:\s*(.+?)\s*(?:\n|$)", re.I|re.S)
A_RE = re.compile(r"answer:\s*(.+?)\s*$", re.I|re.S)

def parse_qa(text: str):
    q = Q_RE.search(text or "")
    a = A_RE.search(text or "")
    return (q.group(1).strip() if q else ""), (a.group(1).strip() if a else "")

def build_text(q, a):
    return f"Question: {q}\nAnswer: {a}"

def tokens(s):  # very light tokenizer
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\.\+\-#]{1,}", (s or "").lower())

def keyword_set(s):
    stop = {"the","a","an","to","of","and","or","in","on","for","with","from","by","at","as","is","are","was","were","it","this","that","these","those"}
    return {t for t in tokens(s) if t not in stop and len(t) >= 3}

def tech_in(s):
    low = (s or "").lower()
    return [t for t in TECH_POOL if t in low]

def make_gibberish_like(s, min_len=12):
    # produce keyboard-like nonsense roughly similar in length
    base = "asdfqwerzxcvtyuiopghjklbnm"
    target = max(min_len, int(len(s) * 0.8))
    out = []
    while len("".join(out)) < target:
        chunk = "".join(random.choice(base) for _ in range(random.randint(3,6)))
        out.append(chunk)
        if random.random() < 0.4:
            out.append(" ")
    g = "".join(out).strip()
    # capitalize a word sometimes to look less uniform
    if random.random() < 0.3:
        g = g.title()
    return g

def replace_tech(answer, banned_words):
    ans_low = answer.lower()
    present = [t for t in TECH_POOL if t in ans_low]
    if not present:
        # insert a wrong tech if none present
        candidate = random.choice([t for t in TECH_POOL if t not in banned_words])
        return answer + f" (implemented with {candidate})"
    wrongs = [t for t in TECH_POOL if t not in present and t not in banned_words]
    if not wrongs:
        return answer
    # replace one present tech with a wrong one
    wrong = random.choice(wrongs)
    target = random.choice(present)
    # naive replace (keep case simple)
    return re.sub(re.escape(target), wrong, answer, flags=re.I)

def hash_key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# --- load -------------------------------------------------------------------
data = json.loads(Path(SRC).read_text())
# keep only items that look like the classifier format
cls = [e for e in data if "input" in e and "label" in e]

pairs = []
for e in cls:
    q, a = parse_qa(e["input"])
    if q and a:
        pairs.append({"q": q, "a": a, "label": e["label"]})

correct = [p for p in pairs if p["label"] == "correct"]
incorrect = [p for p in pairs if p["label"] == "incorrect"]

answers_pool = [p["a"] for p in pairs]

aug = []
seen = set()

def add(q, a, label):
    text = build_text(q, a)
    k = hash_key(text + "|" + label)
    if k not in seen:
        seen.add(k)
        aug.append({"input": text, "label": label})

# keep all original
for p in pairs:
    add(p["q"], p["a"], p["label"])

# create hard negatives from correct examples
for p in correct:
    q, a = p["q"], p["a"]
    q_tokens = keyword_set(q)

    # 1) unrelated: pick an answer that shares few/no keywords with the question
    for _ in range(3):
        ra = random.choice(answers_pool)
        if ra == a:
            continue
        if len(keyword_set(ra) & q_tokens) <= 1:  # low overlap
            add(q, ra, "incorrect")
            break

    # 2) gibberish: similar length keyboard mash
    add(q, make_gibberish_like(a), "incorrect")

    # 3) near-miss: swap tech terms to off-topic ones
    add(q, replace_tech(a, banned_words=q_tokens), "incorrect")

# light downsampling so negatives don’t explode (max 2x positives)
pos_count = sum(1 for e in aug if e["label"] == "correct")
neg = [e for e in aug if e["label"] == "incorrect"]
pos = [e for e in aug if e["label"] == "correct"]

max_neg = min(len(neg), pos_count * 2)
random.shuffle(neg)
final = pos + neg[:max_neg]

random.shuffle(final)
Path(DST).write_text(json.dumps(final, indent=2))
print(f"Saved {len(final)} examples to {DST} (pos={len(pos)}, neg={len(neg[:max_neg])})")
