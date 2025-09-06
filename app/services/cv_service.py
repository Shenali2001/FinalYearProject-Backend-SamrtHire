# app/services/cv_service.py
import os
import re
import json
import time
import random
import logging
import requests
import torch
from typing import Optional, Tuple, List, Dict, Any
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

from transformers import T5Tokenizer, T5ForConditionalGeneration

from app.models.cv import UserCV, Education, Skill, Project, WorkExperience
from app.models.user import User
from app.schemas.cv_details import (
    UserCVCreate, EducationCreate, SkillCreate, ProjectCreate, WorkExperienceCreate
)

# ----------------- Logging -----------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("smart-hire.services.cv_service")

# ----------------- ENV / Gemini -----------------
GEMINI_API_KEY = "AIzaSyB-W42OYrw8EDUAmxyYGyVr1aa1Rs0Mby8"
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash-lite").strip()

def _get_gemini_url() -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    return f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_JUDGE_MODEL}:generateContent?key={GEMINI_API_KEY}"

def _gemini_call(payload: dict) -> Optional[dict]:
    url = _get_gemini_url()
    if not url:
        return None
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code != 200:
            msg = resp.text[:400].replace(GEMINI_API_KEY, "***")
            logger.warning(f"Gemini HTTP {resp.status_code}: {msg}")
            return None
        return resp.json()
    except Exception as e:
        logger.warning(f"Gemini call error: {e}")
        return None

if _get_gemini_url():
    logger.info(f"[judge/feedback] Gemini ENABLED with model='{GEMINI_JUDGE_MODEL}'")
else:
    logger.warning("[judge/feedback] Gemini DISABLED (no GEMINI_API_KEY). Judge & feedback will use local fallback.")

# ----------------- ENV / QG MODEL -----------------
QG_MODEL_PATH = os.getenv("QG_MODEL_PATH", "./technical_qg_enhanced_5986").strip()

# ----------------- QG knobs -----------------
QG_PER_TURN_BUDGET_S = float(os.getenv("QG_PER_TURN_BUDGET_S", "2.8"))
QG_ACCEPT_THRESHOLD  = float(os.getenv("QG_ACCEPT_THRESHOLD", "1.25"))
QG_MIN_WORDS         = int(os.getenv("QG_MIN_WORDS", "8"))
QG_MAX_WORDS         = int(os.getenv("QG_MAX_WORDS", "20"))
QG_MAX_SKILLS_USED   = int(os.getenv("QG_MAX_SKILLS_USED", "16"))
QG_K_PER_PROMPT      = int(os.getenv("QG_K_PER_PROMPT", "6"))
QG_TEMPERATURE       = float(os.getenv("QG_TEMPERATURE", "0.85"))
QG_TOP_P             = float(os.getenv("QG_TOP_P", "0.92"))

# ----------------- Early end threshold -----------------
# Ends interview immediately once score <= END_EARLY_SCORE (default -3)
END_EARLY_SCORE = int(os.getenv("END_EARLY_SCORE", "-3"))

# =========================================================
# Device / lazy-load your QG model
# =========================================================
def _ensure_device(app: FastAPI):
    if not hasattr(app.state, "device") or app.state.device is None:
        dev = ("cuda" if torch.cuda.is_available()
               else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                     else "cpu"))
        app.state.device = torch.device(dev)
        logger.info(f"[QG] Using device: {app.state.device}")

def _ensure_qg_model_loaded(app: FastAPI):
    _ensure_device(app)
    need_load = (
        not hasattr(app.state, "t5_model") or app.state.t5_model is None or
        not hasattr(app.state, "t5_tokenizer") or app.state.t5_tokenizer is None
    )
    if not need_load:
        return
    path = QG_MODEL_PATH or "./question_generation_model"
    try:
        tok = T5Tokenizer.from_pretrained(path)
        model = T5ForConditionalGeneration.from_pretrained(path)
        model.to(app.state.device)
        model.eval()
        app.state.t5_tokenizer = tok
        app.state.t5_model = model
        app.state.qg_model_path = path
        logger.info(f"[QG] Loaded question generator from '{path}'")
    except Exception as e:
        logger.warning(f"[QG] Failed to load '{path}', falling back to 'google/flan-t5-base' — {e}")
        tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        model.to(app.state.device)
        model.eval()
        app.state.t5_tokenizer = tok
        app.state.t5_model = model
        app.state.qg_model_path = "google/flan-t5-base"

# =========================================================
# PDF -> TEXT
# =========================================================
def extract_text_from_pdf_url(pdf_url: str) -> str:
    logger.debug(f"Downloading PDF: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download CV: {e}")

    local_pdf_path = "temp_cv.pdf"
    try:
        with open(local_pdf_path, "wb") as f:
            f.write(response.content)
        import fitz  # PyMuPDF
        doc = fitz.open(local_pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")
    finally:
        try:
            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
        except Exception:
            pass

    return re.sub(r"[ \t]+", " ", text.strip())

# =========================================================
# TEMPLATE VALIDATION + SECTION PARSERS
# =========================================================
def validate_cv_template(text: str) -> bool:
    required = ["SUMMARY:", "EDUCATION:", "SKILLS:", "PROJECTS:", "WORK EXPERIENCE:"]
    up = text.upper()
    return all(s in up for s in required)

def extract_section_text(text: str, section_pattern: str, next_patterns: list) -> str:
    m = re.search(section_pattern + r"\s*(?:\n|$)", text, re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    end = len(text)
    for pat in next_patterns:
        nm = re.search(pat + r"\s*(?:\n|$)", text[start:], re.IGNORECASE)
        if nm:
            end = min(end, start + nm.start())
    return text[start:end].strip()

def parse_education_section(section_text: str) -> list:
    out = []
    if not section_text:
        return out
    for raw in section_text.split("\n"):
        line = raw.strip()
        if not line or not (line.startswith("-") or line.startswith("•")):
            continue
        line = line.strip("-• ").strip()
        parts = [p.strip() for p in line.split(",", 2) if p.strip()]
        if len(parts) >= 2:
            institution = parts[0]
            degree_years = parts[1]
            notes = parts[2] if len(parts) > 2 else ""
            years = re.search(r"(\d{4})\s*-\s*(\d{4}|Present)", degree_years)
            degree = degree_years[: years.start()].strip() if years else degree_years.strip()
            data = {
                "institution": institution,
                "degree": degree,
                "start_year": years.group(1) if years else None,
                "end_year": years.group(2) if years else None,
                "notes": notes,
            }
            try:
                EducationCreate(**data)
                out.append(data)
            except ValueError:
                pass
    return out

def parse_skills_section(section_text: str) -> list:
    out = []
    if not section_text:
        return out
    for raw in section_text.split("\n"):
        line = raw.strip()
        if not line or not (line.startswith("-") or line.startswith("•")):
            continue
        for s in [s.strip() for s in line.strip("-• ").split(",") if s.strip()]:
            try:
                SkillCreate(skill_name=s)
                out.append({"skill_name": s})
            except ValueError:
                pass
    return out

def parse_projects_section(section_text: str) -> list:
    out = []
    if not section_text:
        return out
    for raw in section_text.split("\n"):
        line = raw.strip()
        if not line or not (line.startswith("-") or line.startswith("•")):
            continue
        line = line.strip("-• ").strip()
        if ": " not in line:
            continue
        name, rest = line.split(": ", 1)
        tech_m = re.search(r"Technologies:\s*(.+)", rest, re.IGNORECASE)
        desc = rest[: tech_m.start()].strip(", ") if tech_m else rest.strip(", ")
        tech = tech_m.group(1).strip() if tech_m else None
        data = {"name": name.strip(), "description": desc or None, "technologies": tech}
        try:
            ProjectCreate(**data)
            out.append(data)
        except ValueError:
            pass
    return out

def parse_work_experience_section(section_text: str) -> list:
    out = []
    if not section_text:
        return out
    work: Dict[str, Any] = {}
    desc_lines: List[str] = []
    for raw in section_text.split("\n"):
        line = raw.strip()
        if (line.startswith("-") or line.startswith("•")) and "," in line:
            if work:
                work["description"] = " ".join(desc_lines).strip()
                try:
                    WorkExperienceCreate(**work)
                    out.append(work)
                except ValueError:
                    pass
                desc_lines = []
            parts = line.strip("-• ").split(",", 2)
            company = parts[0].strip()
            role = parts[1].strip() if len(parts) > 1 else ""
            years_desc = parts[2].strip() if len(parts) > 2 else ""
            years = re.search(r"(\d{4})\s*-\s*(\d{4}|Present)", years_desc)
            work = {
                "company": company, "role": role,
                "start_date": years.group(1) if years else None,
                "end_date": years.group(2) if years else None,
                "description": ""
            }
        elif (line.startswith("-") or line.startswith("•")) and work:
            desc_lines.append(line.strip("-• ").strip())
    if work:
        work["description"] = " ".join(desc_lines).strip()
        try:
            WorkExperienceCreate(**work)
            out.append(work)
        except ValueError:
            pass
    return out

def parse_cv_text(text: str) -> dict:
    if not validate_cv_template(text):
        raise ValueError("CV is missing required sections: SUMMARY, EDUCATION, SKILLS, PROJECTS, WORK EXPERIENCE.")
    headers = {
        "summary": r"SUMMARY:",
        "education": r"EDUCATION:",
        "skills": r"SKILLS:",
        "projects": r"PROJECTS:",
        "work_experience": r"WORK EXPERIENCE:",
    }
    pats = list(headers.values())
    return {
        "education": parse_education_section(extract_section_text(text, headers["education"], pats[2:])),
        "skills": parse_skills_section(extract_section_text(text, headers["skills"], pats[3:])),
        "projects": parse_projects_section(extract_section_text(text, headers["projects"], pats[4:])),
        "work_experience": parse_work_experience_section(extract_section_text(text, headers["work_experience"], pats[5:])),
    }

# =========================================================
# DB WRITE
# =========================================================
def create_user_cv_service(user_email: str, cv_data: UserCVCreate, db: Session):
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        cv_text = extract_text_from_pdf_url(cv_data.cv_url)
        parsed = parse_cv_text(cv_text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CV processing failed: {e}")

    user_cv = UserCV(
        user_id=user.id,
        cv_url=cv_data.cv_url,
        job_type_id=cv_data.job_type_id,
        job_position_id=cv_data.job_position_id,
        extracted_text=cv_text,
    )
    db.add(user_cv)
    db.flush()  # get id

    for edu in parsed.get("education", []):
        try:
            db.add(Education(**EducationCreate(**edu).dict(), user_cv_id=user_cv.id))
        except ValueError:
            pass
    for sk in parsed.get("skills", []):
        try:
            db.add(Skill(**SkillCreate(**sk).dict(), user_cv_id=user_cv.id))
        except ValueError:
            pass
    for pj in parsed.get("projects", []):
        try:
            db.add(Project(**ProjectCreate(**pj).dict(), user_cv_id=user_cv.id))
        except ValueError:
            pass
    for wx in parsed.get("work_experience", []):
        try:
            db.add(WorkExperience(**WorkExperienceCreate(**wx).dict(), user_cv_id=user_cv.id))
        except ValueError:
            pass

    db.commit()
    db.refresh(user_cv)
    return user_cv

def get_all_user_cvs_service(db: Session):
    return db.query(UserCV).all()

# =========================================================
# Role & snippets
# =========================================================
def classify_difficulty_from_content(content: str):
    cl = (content or "").lower()
    if any(k in cl for k in ["degree", "education", "distinction", "gpa"]):
        return "easy", "education"
    if any(k in cl for k in ["python", "docker", "react", "spring boot", "springboot", "sql", "mongodb", "java", "javascript", "next.js", "node.js"]):
        return "medium", "skills"
    return "hard", "project"

def _resolve_role_for_user_cv(db: Session, user_cv: UserCV) -> Tuple[str, str]:
    jt, jp = "unknown", "unknown"
    try:
        from app.models.enums import JobType, JobPosition  # type: ignore
        try:
            if getattr(user_cv, "job_type_id", None) is not None:
                jt = JobType(user_cv.job_type_id).name.replace("_", " ").lower()
            if getattr(user_cv, "job_position_id", None) is not None:
                jp = JobPosition(user_cv.job_position_id).name.replace("_", " ").lower()
        except Exception:
            pass
    except Exception:
        pass
    if jt == "unknown" or jp == "unknown":
        try:
            from app.models.job import JobType as JT, JobPosition as JP  # type: ignore
            if jt == "unknown" and getattr(user_cv, "job_type_id", None) is not None:
                rec = db.query(JT).get(user_cv.job_type_id)
                if rec and getattr(rec, "name", None):
                    jt = rec.name.strip().lower()
            if jp == "unknown" and getattr(user_cv, "job_position_id", None) is not None:
                rec = db.query(JP).get(user_cv.job_position_id)
                if rec and getattr(rec, "name", None):
                    jp = rec.name.strip().lower()
        except Exception:
            pass
    return jt or "unknown", jp or "unknown"

def _role_topics(job_type: str, job_position: str) -> List[str]:
    mapping = {
        ("software", "backend engineer"): ["APIs", "Databases", "Concurrency", "Caching", "System design"],
        ("software", "frontend engineer"): ["React", "State", "Performance", "Accessibility", "Testing"],
        ("software", "associate software engineer"): ["Version control", "Unit testing", "Code review", "CI/CD"],
        ("ui/ux", "associate software engineer"): ["User flows", "Wireframes", "Usability", "Prototyping"],
        ("accounting", "financial accountant"): ["GL", "Reconciliation", "ERP", "Reporting", "Compliance"],
    }
    return mapping.get((job_type, job_position), [])

def _infer_topics_from_cv_skills(db: Session, user_cv: UserCV) -> List[str]:
    raw = [s.skill_name or "" for s in db.query(Skill).filter(Skill.user_cv_id == user_cv.id).all()]
    s = " ".join(raw).lower()
    topics: List[str] = []
    def add(k):
        if k not in topics: topics.append(k)
    if any(x in s for x in ["spring", "jpa", "hibernate"]): add("Spring & JPA")
    if any(x in s for x in ["react", "next", "redux"]): add("React & State")
    if any(x in s for x in ["docker", "kubernetes", "k8s"]): add("Containers & Orchestration")
    if any(x in s for x in ["sql", "postgres", "mysql", "mssql"]): add("SQL & Indexing")
    if any(x in s for x in ["mongodb", "redis", "elasticsearch"]): add("NoSQL & Caching")
    if any(x in s for x in ["node", "express", "nestjs", "graphql", "rest"]): add("APIs & Protocols")
    if any(x in s for x in ["aws", "azure", "gcp", "lambda", "s3", "firebase", "firestore"]): add("Cloud & Deployment")
    if any(x in s for x in ["ci/cd", "jenkins", "github actions", "gitlab"]): add("CI/CD & Observability")
    return topics[:5]

def _build_snippets_for_user_cv(db: Session, user_cv: UserCV) -> List[dict]:
    snippets: List[dict] = []
    # SKILLS
    skill_names = []
    seen_low = set()
    for s in db.query(Skill).filter(Skill.user_cv_id == user_cv.id).all():
        name = (s.skill_name or "").strip()
        low = name.lower()
        if name and low not in seen_low:
            seen_low.add(low)
            skill_names.append(name)
    random.shuffle(skill_names)
    for name in skill_names[:QG_MAX_SKILLS_USED]:
        t = f"Skill: {name}."
        d, _ = classify_difficulty_from_content(t)
        snippets.append({"text": t, "difficulty": d})
    # PROJECTS
    for p in db.query(Project).filter(Project.user_cv_id == user_cv.id).all():
        base = f"Project: {p.name}."
        if p.description:
            base += f" {p.description.strip()}"
        if p.technologies:
            base += f" Technologies: {p.technologies}."
        d, _ = classify_difficulty_from_content(base)
        snippets.append({"text": base, "difficulty": d})
    # WORK EXPERIENCE
    for w in db.query(WorkExperience).filter(WorkExperience.user_cv_id == user_cv.id).all():
        span = f"{w.start_date or 'Unknown'}-{w.end_date or 'Present'}"
        desc = (w.description or "").strip()
        if desc:
            t = f"Experience: {w.role} at {w.company} ({span}). Responsibilities: {desc}"
        else:
            t = f"Experience: {w.role} at {w.company} ({span})."
        d, _ = classify_difficulty_from_content(t)
        snippets.append({"text": t, "difficulty": d})
    random.shuffle(snippets)
    return snippets

# =========================================================
# Domain lexicon & coherence guard
# =========================================================
def _tokens_simple(t: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-\+\.#]{1,}", (t or "").lower())]

_DOMAIN_LEX = {
    "db": {
        "sql","postgres","postgresql","mysql","mariadb","mssql","oracle","mongodb","nosql",
        "index","indexes","indexing","transaction","transactions","acid","join","query","queries","jdbc","jpa","hibernate","schema","erd"
    },
    "uiux": {
        "wireframe","wireframes","prototype","prototypes","prototyping","figma","ux","ui","usability",
        "persona","personas","journey","affinity","heuristic","mockup","mockups","low-fidelity","high-fidelity"
    },
    "frontend": {
        "react","next","redux","dom","browser","css","scss","tailwind","html","typescript","vite","webpack","vitejs","spa"
    },
    "backend": {
        "api","apis","grpc","rest","http","spring","springboot","node","express","nestjs","microservice","microservices",
        "jwt","auth","authorization","authentication","controller","service","repository"
    },
    "devops": {
        "docker","kubernetes","k8s","helm","jenkins","github","gitlab","ci","cd","ci/cd","pipeline","pipelines","terraform"
    },
    "cloud": {
        "aws","gcp","azure","lambda","s3","ec2","cloudwatch","cloudrun","cloudsql","appengine",
        "firebase","firestore","realtime","realtime-database","realtimedatabase","cloudfunctions","cloud-function","cloudfunctions",
        "fcm","cloudstorage","firebase-storage","firebaseauth","firebase-auth"
    }
}

_BANNED_PAIRS = {
    frozenset({"uiux","db"}),
    frozenset({"uiux","devops"}),
    frozenset({"uiux","cloud"}),
}
_BAD_CROSS_PHRASES = {
    ("wireframe", "firebase"),
    ("wireframes", "firebase"),
    ("wireframe", "firestore"),
    ("wireframes", "firestore"),
}

def _domains_in(text: str) -> set:
    toks = set(_tokens_simple(text))
    domains = set()
    for d, vocab in _DOMAIN_LEX.items():
        if any(v in toks for v in vocab):
            domains.add(d)
    return domains

def _allowed_domains_for_role(jt: str, jp: str) -> set:
    jt = (jt or "").lower(); jp = (jp or "").lower()
    if jt == "software" and "backend" in jp:
        return {"backend","db","devops","cloud"}
    if jt == "software" and "frontend" in jp:
        return {"frontend","uiux"}
    if "ui/ux" in jt or "uiux" in jt or "ux" in jp or "ui" in jp:
        return {"uiux","frontend"}
    return {"backend","db","frontend","devops","cloud","uiux"}

def _has_bad_phrase_combo(text: str) -> bool:
    tl = (text or "").lower()
    for a, b in _BAD_CROSS_PHRASES:
        if a in tl and b in tl:
            return True
    return False

def _semantic_sanity(text: str) -> bool:
    tl = (text or "").lower()
    if "wireframe" in tl:
        qd = _domains_in(text)
        return qd.issubset({"uiux","frontend"})
    if "erd" in tl or "schema" in tl:
        qd = _domains_in(text)
        return qd.issubset({"db","backend"})
    return True

def _coherence_ok(candidate_q: str, content_keywords: List[str], snippet_domains: set, role_domains: set) -> bool:
    if _has_bad_phrase_combo(candidate_q):
        return False
    if not _semantic_sanity(candidate_q):
        return False
    q_domains = _domains_in(candidate_q)
    toks = set(_tokens_simple(candidate_q))
    has_kw_overlap = any(k.lower() in toks for k in (content_keywords or [])[:6])
    if not has_kw_overlap and not (q_domains & snippet_domains):
        return False
    if q_domains and not q_domains.issubset(role_domains | snippet_domains):
        return False
    for pair in _BANNED_PAIRS:
        if pair.issubset(q_domains):
            return False
    return True

def _preposition_tweaks(text: str) -> str:
    return re.sub(r"\b(at|on)\s+(aws|gcp|azure|firebase|firestore)\b", r"in \2", text, flags=re.IGNORECASE)

# =========================================================
# T5 Question Generation (finetune-style)
# =========================================================
_BAD_META = re.compile(
    r"(ROLE|POSITION|TYPE|SKILLS|CV|CONTEXT|OUTPUT|TEMPLATE|PROMPT|INSTRUCTION|CONSTRAINTS?|QUESTION:|LABELS?|PREAMBLE|METADATA|```|\||^[-*]\s)",
    re.IGNORECASE,
)
_BAD_CHARS = ["|", "`", "•"]
_BANNED_CANON = {
    "role","position","type","skills","cv","context","output","template","prompt",
    "instruction","constraint","question","labels","preamble","metadata","uiux"
}

def _contains_banned_any_spacing(text: str) -> bool:
    canon = re.sub(r"[^a-z]", "", (text or "").lower())
    return any(b in canon for b in _BANNED_CANON)

def _sanitize_question(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip().splitlines()[0].strip(" '\"“”‘’")
    if _contains_banned_any_spacing(text):
        return None
    text = _BAD_META.sub("", text).strip()
    for ch in _BAD_CHARS:
        text = text.replace(ch, " ")
    if text.lower().startswith(("a question", "write", "generate", "create", "compose", "produce")):
        return None
    text = (text.split("?")[0].strip() + "?") if "?" in text else text.rstrip(".!,:; ") + "?"
    text = re.sub(r"\s+", " ", text).strip()
    wc = len(text.split())
    if wc < QG_MIN_WORDS or wc > QG_MAX_WORDS:
        return None
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return re.sub(r"\s+", " ", text).strip()

def _extract_keywords(content: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.]{2,}", content or "")
    toks = [t.strip(",.()").lower() for t in toks
            if t.lower() not in {"and","the","with","for","from","into","using","at","of"}]
    uniq: List[str] = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)
    return uniq[:8]

def _to_ft_prompt_from_content(content: str) -> str:
    txt = (content or "").strip()
    low = txt.lower()
    if low.startswith("skill:"):
        skill = txt.split(":", 1)[1].strip(" .")
        return f"Create a technical interview question about {skill}:\nQuestion:"
    if low.startswith("project:"):
        after = txt.split(":", 1)[1].strip()
        proj = re.split(r"\bTechnologies:\b", after, flags=re.IGNORECASE)[0].strip(" .")
        proj = proj if proj else after
        return f"Generate a project-specific interview question for: {proj}\nQuestion:"
    if low.startswith("experience:"):
        work = txt.split(":", 1)[1].strip(" .")
        return f"Create an experience-based question about: {work}\nQuestion:"
    keywords = _extract_keywords(txt)
    fallback = ", ".join(keywords[:8]) if keywords else txt[:140]
    return f"Generate a technical interview question for: {fallback}\nQuestion:"

@torch.inference_mode()
def _gen_candidates(app: FastAPI, prompt: str, k: int) -> List[str]:
    _ensure_qg_model_loaded(app)
    tok = app.state.t5_tokenizer
    model = app.state.t5_model
    device = app.state.device

    inputs = tok(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
    outs = model.generate(
        **inputs,
        do_sample=True,
        temperature=QG_TEMPERATURE,
        top_p=QG_TOP_P,
        max_new_tokens=40,
        min_new_tokens=12,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        num_return_sequences=max(1, k),
        pad_token_id=tok.pad_token_id,
    )

    seqs = outs.sequences if hasattr(outs, "sequences") else outs
    decoded: List[str] = []
    for seq in seqs:
        q = tok.decode(seq, skip_special_tokens=True).strip()
        q = _sanitize_question(q)
        if q:
            decoded.append(q)

    seen = set(); uniq = []
    for q in decoded:
        key = re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()
        if key not in seen:
            seen.add(key); uniq.append(q)
    return uniq[:k]

# =========================================================
# Scoring helpers
# =========================================================
_TECH_VERBS = {
    "design","implement","optimize","debug","profile","scale","index","partition",
    "replicate","cache","tune","benchmark","deploy","instrument","monitor","refactor",
    "serialize","encrypt","compress","parallelize","shard","paginate"
}
_TECH_HINTS = {
    "api","grpc","rest","http","sql","index","transaction","cache","latency","throughput",
    "concurrency","thread","lock","async","complexity","profiling","memory","gc","jvm",
    "spring","node","react","next","docker","kubernetes","ci/cd","graphql","auth","jwt",
    "sharding","replication","consistency","cap","acid","monitoring","observability","logging",
    "redis","rabbitmq","kafka","elasticsearch","typeorm","hibernate","jpa","postgres","mysql"
}

def _tech_score(q: str, role_topics: List[str], cv_keywords: List[str], q_domains: set, snippet_domains: set, role_domains: set) -> float:
    ql = (q or "").lower()
    score = 0.0
    if any(v in ql for v in _TECH_VERBS): score += 1.05
    if any(h in ql for h in _TECH_HINTS): score += 0.95
    if any((t or "").lower() in ql for t in (role_topics or [])): score += 0.8
    hits = sum(1 for k in (cv_keywords or []) if len(k) >= 3 and k in ql)
    score += min(hits, 3) * 0.45
    wc = len(q.split())
    if QG_MIN_WORDS <= wc <= QG_MAX_WORDS: score += 0.6
    if q.endswith("?"): score += 0.2
    if q_domains and q_domains.issubset(role_domains | snippet_domains): score += 0.5
    for pair in _BANNED_PAIRS:
        if pair.issubset(q_domains): score -= 3.0
    if "this interview" in ql or "at this interview" in ql: score -= 3.0
    if "what performed" in ql: score -= 3.0
    return score

def _canonical(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (q or "").lower()).strip()

def _get_seen_store(app: FastAPI):
    if not hasattr(app.state, "seen_questions"):
        app.state.seen_questions = {}  # {(jt, jp): set(canonical_q)}
    return app.state.seen_questions

@torch.inference_mode()
def _role_aware_generate_one(app: FastAPI, jt: str, jp: str, topics: List[str], content: str) -> Optional[str]:
    keywords = _extract_keywords(content)
    prompts = [
        _to_ft_prompt_from_content(content),
        (f"Generate a technical interview question for: {', '.join(keywords[:6])}\nQuestion:"
         if keywords else None),
    ]
    prompts = [p for p in prompts if p]

    seen_store = _get_seen_store(app)
    seen_key = (jt or "unknown", jp or "unknown")
    seen_store.setdefault(seen_key, set())

    role_domains = _allowed_domains_for_role(jt, jp)
    snippet_domains = _domains_in(content)

    best_q, best_s = None, -1e9
    for p in prompts:
        cands = _gen_candidates(app, p, k=QG_K_PER_PROMPT)
        for q in cands:
            if _canonical(q) in seen_store[seen_key]:
                continue
            if not _coherence_ok(q, keywords, snippet_domains, role_domains):
                continue
            q_domains = _domains_in(q)
            s = _tech_score(q, [t.lower() for t in (topics or [])], keywords, q_domains, snippet_domains, role_domains)
            if s > best_s:
                best_q, best_s = q, s

    if best_q and best_s >= QG_ACCEPT_THRESHOLD:
        best_q = _preposition_tweaks(best_q)
        seen_store[seen_key].add(_canonical(best_q))
        return best_q
    return None

def _text_chunks_for_fallback(text: str, max_chunks: int = 40) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text or "")
    chunks = []
    for p in parts:
        s = re.sub(r"\s+", " ", p).strip()
        if len(s) >= 30:
            chunks.append(s[:400])
        if len(chunks) >= max_chunks:
            break
    return chunks

# =========================================================
# CV selection
# =========================================================
def _pick_best_user_cv_for_email(db: Session, email: str) -> Optional[UserCV]:
    cv_ids = [row[0] for row in db.query(UserCV.id).join(User).filter(User.email == email).all()]
    if not cv_ids:
        return None
    best_id, best_score = None, -1
    for cid in cv_ids:
        edu_n = db.query(Education).filter(Education.user_cv_id == cid).count()
        sk_n  = db.query(Skill).filter(Skill.user_cv_id == cid).count()
        pj_n  = db.query(Project).filter(Project.user_cv_id == cid).count()
        wx_n  = db.query(WorkExperience).filter(WorkExperience.user_cv_id == cid).count()
        score = edu_n + sk_n + pj_n + wx_n
        logger.info(f"[QG] cv_id={cid} counts -> edu={edu_n} skills={sk_n} proj={pj_n} work={wx_n} score={score}")
        if score > best_score or (score == best_score and (best_id is None or cid > best_id)):
            best_score, best_id = score, cid
    return db.query(UserCV).get(best_id) if best_id is not None else None

# =========================================================
# Judge (Gemini + local fallback)
# =========================================================
_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","from","with","by","at","as","is","are","was","were",
    "it","this","that","these","those","be","been","being","i","you","he","she","we","they","my","your","our"
}
def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())
def _tokens(t: str):
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-\+\.#]{1,}", _norm_text(t)) if w not in _STOPWORDS]
def _looks_like_gibberish(answer: str) -> bool:
    a = answer or ""
    if len(a) < 6: return True
    letters = sum(ch.isalpha() for ch in a); total = len(a)
    if total == 0 or (letters / total) < 0.55: return True
    vowel_ratio = (sum(ch in "aeiouAEIOU" for ch in a) / max(1, letters))
    if vowel_ratio < 0.20: return True
    if re.search(r"([a-zA-Z])\1{3,}", a): return True
    toks = _tokens(a); avg_len = (sum(len(t) for t in toks) / max(1, len(toks))) if toks else 0
    return (not toks) or avg_len < 3.0

def _gemini_judge(question: str, answer: str) -> Optional[Tuple[bool, float, str]]:
    rubric = (
        "You are a technical interviewer grading a short answer.\n"
        "Question and Answer are provided as JSON.\n"
        "Mark CORRECT if on-topic AND either:\n"
        "  • includes at least one specific technical detail, OR\n"
        "  • gives a plausible mechanism relevant to the technology.\n"
        "Mark INCORRECT only if off-topic/empty/pure fluff.\n"
        "Return ONLY JSON: {\"is_correct\": true|false, \"confidence\": 0..1, \"reasons\": \"...\"}."
    )
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": rubric}]},
            {"role": "user", "parts": [{"text": json.dumps({"question": question, "answer": answer}, ensure_ascii=False)}]}
        ],
        "generationConfig": {"temperature": 0.15, "topK": 40, "topP": 0.9, "maxOutputTokens": 128}
    }
    data = _gemini_call(body)
    if not data: return None
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.DOTALL).strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m: return None
        obj = json.loads(m.group(0))
        return bool(obj.get("is_correct")), float(obj.get("confidence", 0.0)), str(obj.get("reasons","")).strip()
    except Exception:
        return None

def _local_judge(question: str, answer: str) -> Tuple[bool, float]:
    if _looks_like_gibberish(answer): return False, 0.05
    qk = set(k for k in _tokens(question) if len(k) >= 3)
    ak = set(k for k in _tokens(answer) if len(k) >= 3)
    overlap = len(qk & ak)
    if overlap >= 2: return True, min(0.4 + 0.1 * min(overlap, 4), 0.8)
    return (len(ak) > 8), 0.35 if len(ak) > 8 else 0.2

def _keyword_overlap(question: str, answer: str) -> int:
    q = {w for w in _tokens(question) if len(w) >= 3}
    a = {w for w in _tokens(answer) if len(w) >= 3}
    return len(q & a)

# =========================================================
# Gemini-based detailed FEEDBACK (candidate) & SUMMARY (admin)
# =========================================================
def _gemini_candidate_feedback(history: List[dict], jt: str, jp: str, scorecard: dict) -> Optional[dict]:
    """Ask Gemini for a rich candidate-facing feedback JSON."""
    prompt = (
        "You are a helpful technical interviewer. Input JSON includes role_type, role_position, and history.\n"
        "history is a list of {question, answer, difficulty, is_correct, p_correct}.\n"
        "Write DETAILED candidate feedback as STRICT JSON with keys ONLY:\n"
        "  summary (string),\n"
        "  strengths (array of strings),\n"
        "  areas_to_improve (array of strings),\n"
        "  topic_breakdown (array of {topic, performance('strong'|'ok'|'weak'), evidence}),\n"
        "  next_steps (array of strings),\n"
        "  resources (array of strings, 3-7 items),\n"
        "  suitability ('strong'|'good'|'borderline'|'unsuitable'),\n"
        "  is_suitable (boolean),\n"
        "  overall_rating (number 1-5),\n"
        "  hire_recommendation ('strong_yes'|'yes'|'leaning_no'|'no'),\n"
        "  scorecard (object with existing asked/correct per difficulty and accuracy_pct)."
    )
    payload = {
        "role_type": jt, "role_position": jp,
        "scorecard": scorecard,
        "history": history
    }
    data = _gemini_call({
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
            {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
        ],
        "generationConfig": {"temperature": 0.2, "topK": 40, "topP": 0.9, "maxOutputTokens": 700}
    })
    if not data: return None
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.DOTALL).strip()
        obj = json.loads(re.search(r"\{.*\}", text, flags=re.DOTALL).group(0))
        return obj
    except Exception as e:
        logger.warning(f"[Gemini candidate feedback parse error] {e}")
        return None

def _gemini_admin_summary(history: List[dict], jt: str, jp: str, scorecard: dict) -> Optional[dict]:
    """Ask Gemini for a concise HR/admin summary JSON."""
    prompt = (
        "You are a hiring panel summarizer for HR. Input JSON includes role_type, role_position, scorecard, and history.\n"
        "Return STRICT JSON with keys ONLY:\n"
        "  overview (string, 2-4 sentences),\n"
        "  competency_scores (object with numbers 0-5): {problem_solving, coding, system_design, databases, web_backend, devops_testing},\n"
        "  notable_signals (array of strings),\n"
        "  risk_flags (array of strings),\n"
        "  recommendation (object): {decision('advance'|'onsite'|'hold'|'reject'), confidence(0..1), level('junior'|'mid'|'senior')},\n"
        "  followups (array of strings),\n"
        "  question_log (array of {question, difficulty, is_correct, brief_reason})."
    )
    payload = {
        "role_type": jt, "role_position": jp,
        "scorecard": scorecard,
        "history": history
    }
    data = _gemini_call({
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
            {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
        ],
        "generationConfig": {"temperature": 0.2, "topK": 40, "topP": 0.9, "maxOutputTokens": 700}
    })
    if not data: return None
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.DOTALL).strip()
        obj = json.loads(re.search(r"\{.*\}", text, flags=re.DOTALL).group(0))
        return obj
    except Exception as e:
        logger.warning(f"[Gemini admin summary parse error] {e}")
        return None

# ----------------- Local fallbacks for richer outputs -----------------
def _local_scorecard(state: dict) -> dict:
    e_asked = state["per_difficulty"]["easy"]["asked"]
    e_corr  = state["per_difficulty"]["easy"]["correct"]
    m_asked = state["per_difficulty"]["medium"]["asked"]
    m_corr  = state["per_difficulty"]["medium"]["correct"]
    h_asked = state["per_difficulty"]["hard"]["asked"]
    h_corr  = state["per_difficulty"]["hard"]["correct"]
    asked = e_asked + m_asked + h_asked
    corr  = e_corr  + m_corr  + h_corr
    acc = (corr / asked * 100.0) if asked else 0.0
    return {
        "easy":   {"asked": e_asked, "correct": e_corr},
        "medium": {"asked": m_asked, "correct": m_corr},
        "hard":   {"asked": h_asked, "correct": h_corr},
        "accuracy_pct": round(acc, 1)
    }

def _local_candidate_feedback(history: List[dict], scorecard: dict) -> dict:
    asked = sum(1 for _ in history)
    corr  = sum(1 for h in history if h.get("is_correct"))
    acc = (corr / asked * 100.0) if asked else 0.0
    suitability = "strong" if acc >= 75 else "good" if acc >= 55 else "borderline" if acc >= 35 else "unsuitable"
    is_suitable = acc >= 55
    strengths, gaps, nexts, resources = [], [], [], []
    if corr >= 2:
        strengths.append("Showed understanding of at least a couple of core topics.")
    if scorecard["hard"]["correct"] >= 1:
        strengths.append("Handled an advanced topic under pressure.")
    if (scorecard["medium"]["asked"] + scorecard["hard"]["asked"]) - (scorecard["medium"]["correct"] + scorecard["hard"]["correct"]) >= 2:
        gaps.append("Needs deeper coverage of mid/advanced topics (APIs, DB indexing, performance).")
        nexts.append("Practice small system design prompts and database indexing scenarios.")
    if scorecard["easy"]["asked"] > scorecard["easy"]["correct"]:
        gaps.append("Revisit fundamentals to avoid basic slips.")
        nexts.append("Review language basics and common standard libraries.")
    resources = [
        "Designing Data-Intensive Applications (Martin Kleppmann)",
        "System Design Primer (GitHub)",
        "PostgreSQL Query Planning & Indexing docs",
        "gRPC/REST API best practices (Microsoft/Google style guides)",
        "Production-Grade Docker & Kubernetes (tutorial series)"
    ]
    topic_breakdown = []
    for h in history[-5:]:
        topic_breakdown.append({
            "topic": "recent_question",
            "performance": "strong" if h.get("is_correct") else "weak",
            "evidence": (h.get("answer") or "")[:140]
        })
    return {
        "summary": "Automated local feedback (Gemini unavailable).",
        "strengths": strengths,
        "areas_to_improve": gaps if gaps else ["Add more concrete examples in answers."],
        "topic_breakdown": topic_breakdown,
        "next_steps": nexts if nexts else ["Attempt timed practice on role-relevant topics."],
        "resources": resources[:5],
        "suitability": suitability,
        "is_suitable": is_suitable,
        "overall_rating": round(acc/25, 1),  # 0..4 -> approx 0..100%
        "hire_recommendation": "yes" if is_suitable else "no",
        "scorecard": scorecard
    }

def _local_admin_summary(history: List[dict], scorecard: dict) -> dict:
    asked = len(history)
    corr  = sum(1 for h in history if h.get("is_correct"))
    acc   = (corr / asked * 100.0) if asked else 0.0
    decision = "onsite" if acc >= 65 else "hold" if acc >= 50 else "reject"
    level = "senior" if acc >= 80 else "mid" if acc >= 55 else "junior"
    return {
        "overview": "Automated local admin summary (Gemini unavailable). Candidate showed mixed performance; see scorecard and question log.",
        "competency_scores": {
            "problem_solving": round(min(5.0, 1.5 + acc/20), 1),
            "coding": round(min(5.0, 1.2 + acc/25), 1),
            "system_design": round(min(5.0, 0.8 + acc/30), 1),
            "databases": round(min(5.0, 0.8 + acc/28), 1),
            "web_backend": round(min(5.0, 1.0 + acc/26), 1),
            "devops_testing": round(min(5.0, 0.6 + acc/35), 1),
        },
        "notable_signals": ["See recent answers for evidence; candidate accuracy trends are summarized in the scorecard."],
        "risk_flags": ["Knowledge gaps on mid/advanced topics"] if acc < 55 else [],
        "recommendation": {"decision": decision, "confidence": round(min(1.0, 0.4 + acc/100), 2), "level": level},
        "followups": ["Probe system design trade-offs", "Deep dive on DB indexing and query plans"],
        "question_log": [
            {
                "question": h.get("question","")[:180],
                "difficulty": h.get("difficulty",""),
                "is_correct": bool(h.get("is_correct")),
                "brief_reason": "On-topic" if h.get("is_correct") else "Insufficient mechanism/detail"
            } for h in history
        ]
    }

# =========================================================
# Build final combined outputs (candidate + admin) and persist
# =========================================================
def _build_final_reports(state: dict, jt: str, jp: str) -> dict:
    # Build history snapshot
    hist = [
        {
            "question": h.get("question",""),
            "answer": h.get("answer",""),
            "difficulty": h.get("difficulty","medium"),
            "is_correct": bool(h.get("is_correct")),
            "p_correct": float(h.get("p_correct", 0.0)),
        } for h in state.get("history", [])
    ]
    scorecard = _local_scorecard(state)

    # Try Gemini for both outputs
    cand_fb = _gemini_candidate_feedback(hist, jt, jp, scorecard) or _local_candidate_feedback(hist, scorecard)
    admin_sum = _gemini_admin_summary(hist, jt, jp, scorecard) or _local_admin_summary(hist, scorecard)

    return {"feedback": cand_fb, "admin_summary": admin_sum}

def _persist_final_report(db: Session, email: str, state: dict, final: dict) -> Optional[int]:
    """
    Save the end-of-interview feedback to DB. Returns report id (or None).
    Requires app.models.interview.InterviewReport to exist.
    """
    try:
        from app.models.interview_reports import InterviewReport
    except Exception as e:
        logger.warning(f"[report] model import failed: {e}")
        return None

    try:
        user = db.query(User).filter(User.email == email).first()
        user_id = user.id if user else None

        user_cv = None
        try:
            user_cv = _pick_best_user_cv_for_email(db, email)
        except Exception:
            pass
        user_cv_id = user_cv.id if user_cv else None

        cand_fb = (final or {}).get("feedback") or {}
        admin_sum = (final or {}).get("admin_summary") or {}
        scorecard = cand_fb.get("scorecard", {})

        # Accuracy for convenience if Gemini omitted it
        if "accuracy_pct" not in scorecard:
            e_asked = state["per_difficulty"]["easy"]["asked"]
            e_corr  = state["per_difficulty"]["easy"]["correct"]
            m_asked = state["per_difficulty"]["medium"]["asked"]
            m_corr  = state["per_difficulty"]["medium"]["correct"]
            h_asked = state["per_difficulty"]["hard"]["asked"]
            h_corr  = state["per_difficulty"]["hard"]["correct"]
            asked = e_asked + m_asked + h_asked
            corr  = e_corr  + m_corr  + h_corr
            scorecard["accuracy_pct"] = round((corr / asked) * 100.0, 1) if asked else 0.0

        # Persist using existing columns; store admin_summary in raw_feedback for now
        rec = InterviewReport(
            user_id=user_id,
            user_cv_id=user_cv_id,
            email=email,
            role_type=state.get("role_type"),
            role_position=state.get("role_position"),
            status=final.get("status") or "finished",
            score=int(final.get("score") or 0),
            questions_asked=int(final.get("questions_asked") or 0),
            accuracy_pct=float(scorecard.get("accuracy_pct") or 0.0),
            suitability=cand_fb.get("suitability"),
            is_suitable=bool(cand_fb.get("is_suitable", False)),
            summary=cand_fb.get("summary"),  # candidate-facing summary
            strengths=cand_fb.get("strengths"),
            areas_to_improve=cand_fb.get("areas_to_improve"),
            next_steps=cand_fb.get("next_steps"),
            scorecard=scorecard,
            history=state.get("history"),
            raw_feedback={
                "ended_by": final.get("ended_by"),
                "candidate": cand_fb,
                "admin": admin_sum
            },
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        logger.info(f"[report] saved InterviewReport id={rec.id} email={email}")
        return rec.id
    except Exception as e:
        logger.warning(f"[report] save failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return None

# =========================================================
# On-demand NEXT question generation (no pre-pool)
# =========================================================
def _prepare_context(email: str, db: Session) -> Tuple[str, str, List[str], List[dict], List[str]]:
    user_cv = _pick_best_user_cv_for_email(db, email)
    if not user_cv:
        raise HTTPException(status_code=404, detail="CV not found for user")
    jt, jp = _resolve_role_for_user_cv(db, user_cv)
    topics = _role_topics(jt, jp) or _infer_topics_from_cv_skills(db, user_cv)
    snippets = _build_snippets_for_user_cv(db, user_cv)
    chunks = _text_chunks_for_fallback(user_cv.extracted_text or "", max_chunks=60)
    return jt, jp, topics, snippets, chunks

def _next_question_on_demand(state: dict, app: FastAPI) -> Optional[dict]:
    t0 = time.perf_counter()
    jt = state.get("role_type") or "unknown"
    jp = state.get("role_position") or "unknown"
    topics = state.get("topics") or []
    try_order = [state.get("current_difficulty","medium"), "medium", "easy", "hard"]

    for diff in try_order:
        bucket = state["snippet_buckets"].get(diff, [])
        while bucket and (time.perf_counter() - t0) < QG_PER_TURN_BUDGET_S:
            idx = bucket.pop(0)
            if idx in state["used_snippet_idx"]:
                continue
            state["used_snippet_idx"].add(idx)
            content = state["snippets"][idx]["text"]
            q = _role_aware_generate_one(app, jt, jp, topics, content)
            if q:
                return {"question": q, "difficulty": diff, "scored": True}

    while state["chunk_cursor"] < len(state["chunks"]) and (time.perf_counter() - t0) < QG_PER_TURN_BUDGET_S:
        content = state["chunks"][state["chunk_cursor"]]
        state["chunk_cursor"] += 1
        q = _role_aware_generate_one(app, jt, jp, topics, content)
        if q:
            return {"question": q, "difficulty": "medium", "scored": True}

    logger.info(f"[QG] NEXT could not generate within ~{QG_PER_TURN_BUDGET_S}s.")
    return None

# =========================================================
# Interview flow
# =========================================================
ICEBREAKER_QUESTION = (
    "To warm up, tell me about yourself — your background, key strengths, and what you’re looking for next."
)

def _init_interviews_store(app: FastAPI):
    if not hasattr(app.state, "interviews"):
        app.state.interviews: Dict[str, dict] = {}

def start_adaptive_interview_service(email: str, db: Session, app: FastAPI, max_questions: int = 10):
    _init_interviews_store(app)
    if email in app.state.interviews:
        app.state.interviews.pop(email, None)

    jt, jp, topics, snippets, chunks = _prepare_context(email, db)

    snippet_buckets = {"easy": [], "medium": [], "hard": []}
    for i, s in enumerate(snippets):
        snippet_buckets.setdefault(s["difficulty"], []).append(i)

    state = {
        "asked": [],
        "asked_count": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "per_difficulty": {
            "easy":   {"asked": 0, "correct": 0},
            "medium": {"asked": 0, "correct": 0},
            "hard":   {"asked": 0, "correct": 0},
        },
        "history": [],
        "last": None,
        "score": 0,
        "max_questions": max_questions,
        "current_difficulty": "medium",
        "stopped": False,
        "phase": "PRELUDE",
        "icebreaker": ICEBREAKER_QUESTION,
        "icebreaker_answer": None,

        "role_type": jt,
        "role_position": jp,
        "topics": topics,
        "snippets": snippets,
        "snippet_buckets": snippet_buckets,
        "used_snippet_idx": set(),
        "chunks": chunks,
        "chunk_cursor": 0,
    }
    app.state.interviews[email] = state

    return {"question": state["icebreaker"], "difficulty": "none", "tag": "icebreaker", "scored": False}

def _pop_and_mark(state: dict, qdict: dict):
    q = qdict["question"]; d = qdict["difficulty"]
    if q not in state["asked"]:
        state["asked"].append(q)
        state["asked_count"] += 1
        state["per_difficulty"][d]["asked"] += 1
        state["last"] = {"question": q, "difficulty": d}

def evaluate_answer_for_question(question: str, answer: str, app: FastAPI) -> Tuple[bool, float]:
    q = (question or "").strip()
    a = (answer or "").strip()
    if _looks_like_gibberish(a):
        logger.debug("Answer rejected by gibberish gate.")
        return False, 0.0
    judged = _gemini_judge(q, a)
    if judged is None:  # local fallback
        return _local_judge(q, a)
    is_corr, conf, _ = judged
    if (not is_corr) and conf <= 0.35 and _keyword_overlap(q, a) >= 1:
        logger.info("[Judge=Override] Low-conf negative but on-topic -> accept")
        return True, 0.55
    return is_corr, conf

def submit_adaptive_answer_service(email: str, question: Optional[str], answer: str, app: FastAPI, db: Session):
    _init_interviews_store(app)
    state = app.state.interviews.get(email)
    if not state or state.get("stopped"):
        raise HTTPException(status_code=400, detail="No active interview for user or interview already ended")

    # PRELUDE
    if state.get("phase") == "PRELUDE":
        state["icebreaker_answer"] = answer
        state["phase"] = "MAIN"

        nxt = _next_question_on_demand(state, app)
        if not nxt:
            state["stopped"] = True
            jt = state.get("role_type") or "unknown"
            jp = state.get("role_position") or "unknown"
            final = {"status": "finished", "score": state["score"], "questions_asked": state["asked_count"]}
            reports = _build_final_reports(state, jt, jp)
            final.update(reports)
            report_id = _persist_final_report(db, email, state, final)
            if report_id:
                final["report_id"] = report_id
            app.state.interviews.pop(email, None)
            return {"result": True, "next": None, "final": final}

        _pop_and_mark(state, nxt)
        return {"result": True, "next": nxt, "final": None}

    # MAIN
    last = state.get("last") or {}
    q_text = last.get("question") or (question or "")
    q_diff = last.get("difficulty") or state.get("current_difficulty") or "medium"

    is_correct, confidence = evaluate_answer_for_question(q_text, answer, app)
    state["history"].append({
        "question": q_text, "answer": answer, "difficulty": q_diff,
        "is_correct": is_correct, "p_correct": confidence
    })

    if is_correct:
        state["correct_count"] += 1
        state["per_difficulty"][q_diff]["correct"] += 1
        state["score"] += 1
        if state["current_difficulty"] == "easy":
            state["current_difficulty"] = "medium"
        elif state["current_difficulty"] == "medium":
            state["current_difficulty"] = "hard"
    else:
        state["incorrect_count"] += 1
        state["score"] -= 1
        if state["current_difficulty"] == "hard":
            state["current_difficulty"] = "medium"
        elif state["current_difficulty"] == "medium":
            state["current_difficulty"] = "easy"

    # ---------- EARLY TERMINATION BY SCORE ----------
    if state["score"] <= END_EARLY_SCORE:
        state["stopped"] = True
        jt = state.get("role_type") or "unknown"
        jp = state.get("role_position") or "unknown"
        final = {
            "status": "finished",
            "ended_by": "score_threshold",
            "score": state["score"],
            "questions_asked": state["asked_count"]
        }
        reports = _build_final_reports(state, jt, jp)
        final.update(reports)
        report_id = _persist_final_report(db, email, state, final)
        if report_id:
            final["report_id"] = report_id
        app.state.interviews.pop(email, None)
        return {"result": is_correct, "next": None, "final": final}
    # -----------------------------------------------

    if state["asked_count"] >= state["max_questions"]:
        state["stopped"] = True
        jt = state.get("role_type") or "unknown"
        jp = state.get("role_position") or "unknown"
        final = {"status": "finished", "score": state["score"], "questions_asked": state["asked_count"]}
        reports = _build_final_reports(state, jt, jp)
        final.update(reports)
        report_id = _persist_final_report(db, email, state, final)
        if report_id:
            final["report_id"] = report_id
        app.state.interviews.pop(email, None)
        return {"result": is_correct, "next": None, "final": final}

    nxt = _next_question_on_demand(state, app)
    if not nxt:
        state["stopped"] = True
        jt = state.get("role_type") or "unknown"
        jp = state.get("role_position") or "unknown"
        final = {"status": "finished", "score": state["score"], "questions_asked": state["asked_count"]}
        reports = _build_final_reports(state, jt, jp)
        final.update(reports)
        report_id = _persist_final_report(db, email, state, final)
        if report_id:
            final["report_id"] = report_id
        app.state.interviews.pop(email, None)
        return {"result": is_correct, "next": None, "final": final}

    _pop_and_mark(state, nxt)
    return {"result": is_correct, "next": nxt, "final": None}
