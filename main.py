import json, re, base64, asyncio
from statistics import mean, median, pstdev, pvariance, mode
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import config

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

HEAD = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}", "Content-Type": "application/json"}


async def chat(messages, model=None, max_tokens=1500, retries=3):
    body = {"model": model or config.TEXT_MODEL, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            try:
                r = await c.post(f"{config.AIPIPE_BASE}/chat/completions", headers=HEAD, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = str(e)
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")


GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]


async def gemini_transcribe(payload, attempts_per_model=3):
    last_err = ""
    async with httpx.AsyncClient(timeout=120) as c:
        for model in GEMINI_MODELS:
            for attempt in range(attempts_per_model):
                try:
                    r = await c.post(
                        f"https://aipipe.org/geminiv1beta/models/{model}:generateContent",
                        headers={"Authorization": f"Bearer {config.AIPIPE_TOKEN}"},
                        json=payload)
                    if r.status_code in (429, 500, 502, 503, 504):
                        last_err = f"HTTP {r.status_code} on {model}: {r.text[:160]}"
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = r.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError):
                    last_err = f"empty candidates on {model}"
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__} on {model}: {str(e)[:160]}"
                    await asyncio.sleep(1.0 * (attempt + 1))
    global last_debug_info
    last_debug_info["transcribe_error"] = last_err
    return ""


def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


last_debug_info = {}


@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}


@app.get("/debug")
def get_debug():
    return last_debug_info


def _find_audio_b64(body):
    audio_id, audio_b64 = None, ""
    if isinstance(body, dict):
        for k, v in body.items():
            lk = str(k).lower()
            if isinstance(v, str):
                if ("audio" in lk or "data" in lk or "b64" in lk or "base64" in lk) and len(v) > 200:
                    if len(v) > len(audio_b64):
                        audio_b64 = v
                elif "id" in lk and not audio_id:
                    audio_id = v
    return audio_id, audio_b64


@app.post("/answer-audio")
async def answer_audio(request: Request):
    global last_debug_info
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    last_debug_info = {"content_type": ctype, "raw_len": len(raw)}

    body, audio_id, audio_b64 = {}, None, ""
    try:
        if "application/json" in ctype or raw[:1] in (b"{", b"["):
            body = json.loads(raw)
            last_debug_info["body_keys"] = list(body.keys()) if isinstance(body, dict) else "non-dict"
            audio_id, audio_b64 = _find_audio_b64(body)
        else:
            try:
                form = await request.form()
                for k, v in form.items():
                    data = await v.read() if hasattr(v, "read") else None
                    if data:
                        audio_b64 = base64.b64encode(data).decode()
            except Exception:
                pass
            if not audio_b64 and raw:
                audio_b64 = base64.b64encode(raw).decode()
    except Exception as e:
        last_debug_info["parse_error"] = str(e)

    last_debug_info["body_id"] = audio_id
    last_debug_info["audio_b64_len"] = len(audio_b64)

    transcript = ""
    try:
        audio = base64.b64decode(audio_b64) if audio_b64 else b""
        last_debug_info["magic_bytes"] = audio[:16].hex()

        if audio.startswith(b"ID3") or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            mime = "audio/mp3"
        elif audio.startswith(b"OggS"):
            mime = "audio/ogg"
        elif audio.startswith(b"fLaC"):
            mime = "audio/flac"
        elif audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
            mime = "audio/wav"
        elif audio.startswith(b"\x1aE\xdf\xa3"):
            mime = "audio/webm"
        elif audio[4:8] == b"ftyp":
            mime = "audio/mp4"
        else:
            mime = "audio/wav"
        last_debug_info["detected_mime"] = mime

        payload = {
            "contents": [{
                "parts": [
                    {"text": "Transcribe this audio precisely in Korean. Output ONLY the Korean transcription, nothing else."},
                    {"inlineData": {"mimeType": mime, "data": audio_b64}}
                ]
            }]
        }
        transcript = await gemini_transcribe(payload)
    except Exception as e:
        last_debug_info["exception"] = str(e)

    last_debug_info["transcript"] = transcript

    prompt = (
        "The transcript (Korean) describes a tabular dataset and asks for or states specific statistics. "
        "Extract the raw data, schema, and identify/extract the exact statistics.\n"
        "If the transcript only ASKS to generate data (e.g., 'Generate 140 rows. The median of income is 45000'), do NOT invent data. "
        "Instead, extract the column names into 'columns', return the requested number of rows in 'num_rows', and leave 'data_rows' empty. "
        "ALSO, if it explicitly mentions any constraints or known statistical values (like mean, median, value ranges or allowed values), extract them into 'explicit_stats'.\n\n"
        "Korean to English Statistic Mapping Guide:\n"
        "- '평균' -> 'mean'\n- '표준편차' -> 'std'\n- '분산' -> 'variance'\n"
        "- '최소' / '최솟값' -> 'min'\n- '최대' / '최댓값' -> 'max'\n"
        "- '중앙값' / '중간값' -> 'median'\n- '최빈값' -> 'mode'\n- '범위' -> 'range'\n"
        "- '~사이' (between A and B) -> 'value_range'\n"
        "- '허용값' / '허용된 값' -> 'allowed_values'\n"
        "- '상관관계' -> 'correlation' ('양의'/비례 = positive, '음의'/반비례 = negative)\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"columns\": [\"column_name\"],\n"
        "  \"data_rows\": [[val1], [val2], ...],\n"
        "  \"num_rows\": 140,\n"
        "  \"explicit_stats\": {\n"
        "    \"value_range\": {\"점수\": [0, 100]},\n"
        "    \"median\": {\"소득\": 45000},\n"
        "    \"mean\": {\"온도\": 22},\n"
        "    \"std\": {\"온도\": 3},\n"
        "    \"correlation\": [{\"x\": \"키\", \"y\": \"몸무게\", \"type\": \"positive\"}]\n"
        "  },\n"
        "  \"requested_stats\": [\"median\"]\n"
        "}\n"
        "CRITICAL RULES:\n"
        "1. DO NOT confuse '중간값'/'중앙값' (median) with '평균' (mean).\n"
        "2. DO NOT invent data. Extract all rows exactly as dictated.\n"
        "3. Keep column names exactly as spoken.\n"
        "4. allowed_values is for CATEGORICAL columns whose text explicitly lists a fixed permitted set "
        "('<col>는/은 A, B, C 중 하나입니다', '허용값'/'허용된 값'). In those cases emit "
        "explicit_stats.allowed_values={\"<col>\": [\"A\",\"B\",\"C\"]} AND put <col> in 'columns' AND "
        "put 'allowed_values' in requested_stats. For purely numeric columns with NO listed category set, "
        "NEVER emit allowed_values.\n"
        "5. correlation MUST be a LIST of objects {\"x\": colA, \"y\": colB, \"type\": \"positive\"|\"negative\"}.\n"
        "6. If the transcript states a constraint like '값은 0에서 1 사이입니다', extract the subject as the "
        "column name into 'columns', AND map the constraint in 'explicit_stats'.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    columns, data_rows, req_stats, num_rows, explicit_stats = [], [], [], None, {}
    try:
        raw_llm = await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500)
        last_debug_info["raw_llm"] = raw_llm
        ext = parse_json(raw_llm)
        columns = ext.get("columns", []) or []
        data_rows = ext.get("data_rows", []) or []
        req_stats = ext.get("requested_stats", [])
        num_rows = ext.get("num_rows")
        explicit_stats = ext.get("explicit_stats", {})
    except Exception as e:
        last_debug_info["llm_exception"] = str(e)

    def _extract_allowed_values(tr):
        found = {}
        if not tr:
            return found
        for m in re.finditer(r"([가-힣A-Za-z0-9_]+?)(?:는|은|이|가)\s+([^.。\n]+?)\s*중\s*(?:하나|에서)", tr):
            col = m.group(1).strip()
            vals = [v.strip() for v in re.split(r"[,、/]|또는|혹은", m.group(2)) if v.strip()]
            if col and len(vals) >= 2:
                found[col] = vals
        for m in re.finditer(r"([가-힣A-Za-z0-9_]+?)(?:의|는|은)?\s*허용(?:값|된\s*값)[은는]?\s*[:：]?\s*([^.。\n]+)", tr):
            col = m.group(1).strip()
            rawv = re.sub(r"(입니다|이다)\s*$", "", m.group(2).strip())
            vals = [v.strip() for v in re.split(r"[,、/]|또는|혹은", rawv) if v.strip()]
            if col and vals:
                found[col] = vals
        return found

    av = _extract_allowed_values(transcript)
    if av:
        es_av = explicit_stats.setdefault("allowed_values", {})
        for col, vals in av.items():
            es_av.setdefault(col, vals)
        full_set = {"mean", "std", "variance", "min", "max", "median", "mode",
                    "range", "allowed_values", "value_range", "correlation"}
        if "allowed_values" not in req_stats and set(req_stats) != full_set:
            req_stats.append("allowed_values")

    referenced = []
    for sd in (explicit_stats or {}).values():
        if isinstance(sd, dict):
            for k in sd:
                if k not in referenced:
                    referenced.append(k)
    for c in referenced:
        if c not in columns:
            columns.append(c)

    if not req_stats:
        req_stats = ["mean", "std", "variance", "min", "max", "median", "mode",
                      "range", "allowed_values", "value_range", "correlation"]

    actual_rows = num_rows if num_rows is not None else len(data_rows)
    out = {"rows": actual_rows, "columns": columns,
           "mean": {}, "std": {}, "variance": {}, "min": {}, "max": {},
           "median": {}, "mode": {}, "range": {}, "allowed_values": {},
           "value_range": {}, "correlation": []}

    def col_values(ci):
        vals = []
        for r in data_rows:
            try:
                vals.append(float(r[ci]))
            except Exception:
                pass
        return vals

    cols_vals = []
    for ci, name in enumerate(columns):
        v = col_values(ci)
        if not v:
            continue
        cols_vals.append(v)
        if "mean" in req_stats: out["mean"][name] = mean(v)
        if "std" in req_stats: out["std"][name] = pstdev(v) if len(v) > 1 else 0.0
        if "variance" in req_stats: out["variance"][name] = pvariance(v) if len(v) > 1 else 0.0
        if "min" in req_stats: out["min"][name] = min(v)
        if "max" in req_stats: out["max"][name] = max(v)
        if "median" in req_stats: out["median"][name] = median(v)
        if "mode" in req_stats:
            try: out["mode"][name] = mode(v)
            except Exception: out["mode"][name] = v[0]
        if "range" in req_stats: out["range"][name] = max(v) - min(v)
        if "value_range" in req_stats: out["value_range"][name] = [min(v), max(v)]

    def _corr_type(tr, hint=""):
        h = str(hint).lower()
        if h in ("positive", "negative"):
            return h
        t = (tr or "")
        if "음의" in t or "반비례" in t or "negative" in t.lower():
            return "negative"
        return "positive"

    corr_list = []
    raw_corr = explicit_stats.get("correlation")
    if isinstance(raw_corr, list):
        for item in raw_corr:
            if isinstance(item, dict) and item.get("x") and item.get("y"):
                corr_list.append({"x": item["x"], "y": item["y"],
                                   "type": _corr_type(transcript, item.get("type", ""))})
    elif isinstance(raw_corr, dict):
        for x, y in raw_corr.items():
            if isinstance(y, str) and y:
                corr_list.append({"x": x, "y": y, "type": _corr_type(transcript)})
    if not corr_list and cols_vals and len(columns) > 1 and all(cols_vals) and "correlation" in req_stats:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                a, b = cols_vals[i], cols_vals[j]
                if len(a) == len(b) and len(a) > 1:
                    ma, mb = mean(a), mean(b)
                    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                    corr_list.append({"x": columns[i], "y": columns[j],
                                       "type": "negative" if num < 0 else "positive"})
    if corr_list:
        out["correlation"] = corr_list

    FULL = ["mean", "std", "variance", "min", "max", "median", "mode",
            "range", "allowed_values", "value_range", "correlation"]
    has_data = len(data_rows) > 0

    def _present(s):
        v = explicit_stats.get(s)
        return (isinstance(v, dict) and bool(v)) or (isinstance(v, list) and bool(v))

    if req_stats and set(req_stats) != set(FULL):
        target = [s for s in FULL if s in req_stats]
    elif has_data:
        target = list(FULL)
    else:
        target = [s for s in FULL if _present(s)]

    vr = explicit_stats.get("value_range")
    if isinstance(vr, dict):
        for col, bounds in vr.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo, hi = bounds[0], bounds[1]
                if "min" in target: explicit_stats.setdefault("min", {}).setdefault(col, lo)
                if "max" in target: explicit_stats.setdefault("max", {}).setdefault(col, hi)
                if "range" in target:
                    try: explicit_stats.setdefault("range", {}).setdefault(col, hi - lo)
                    except Exception: pass
    emin, emax = explicit_stats.get("min"), explicit_stats.get("max")
    if isinstance(emin, dict) and isinstance(emax, dict):
        for col in emin:
            if col in emax:
                if "value_range" in target:
                    explicit_stats.setdefault("value_range", {}).setdefault(col, [emin[col], emax[col]])
                if "range" in target:
                    try: explicit_stats.setdefault("range", {}).setdefault(col, emax[col] - emin[col])
                    except Exception: pass

    for stat_name, stat_dict in explicit_stats.items():
        if stat_name in out and isinstance(out[stat_name], dict) and isinstance(stat_dict, dict):
            out[stat_name].update(stat_dict)

    for k in FULL:
        if k == "correlation":
            continue
        if k not in target:
            out[k] = {}
    if "correlation" not in target:
        out["correlation"] = []

    last_debug_info["requested_stats"] = req_stats
    last_debug_info["target_keys"] = target
    last_debug_info["answer"] = out

    return out
