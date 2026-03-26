import os
import json
import base64
import re
from io import BytesIO

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- PDF Generation Imports ---
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

app = FastAPI()

def get_ai_client():
    key = os.getenv("GEMINI_LIVE_KEY", os.getenv("GEMINI_API_KEY", "MISSING_KEY"))
    return genai.Client(api_key=key)

# ==========================================
# 1. AI MODELS
# ==========================================
class BrainDumpRequest(BaseModel):
    transcript: str

class Experience(BaseModel):
    company: str
    role: str
    startMonth: str
    startYear: str
    endMonth: str
    endYear: str
    bullets: str

class Project(BaseModel):
    name: str
    startMonth: str
    startYear: str
    endMonth: str
    endYear: str
    bullets: str

# ✨ FIXED: Added Personal Info Fields so the AI remembers your name/about!
class ResumeExtraction(BaseModel):
    reply: str
    first_name: str = Field(default="")
    last_name: str = Field(default="")
    target_role: str = Field(default="")
    summary: str = Field(default="")
    skills_suggested: list[str]
    experience: list[Experience]
    projects: list[Project]
    missing_fields: list[str]

class CoverLetterRequest(BaseModel):
    job_description: str
    vault_data: str

class CoverLetterResponse(BaseModel):
    cover_letter: str

class AnalyticsRequest(BaseModel):
    vault_data: str
    target_role: str

class AnalyticsResponse(BaseModel):
    strengths: list[str]
    gaps: list[str]

class InterviewRequest(BaseModel):
    target_role: str
    job_description: str
    vault_data: str

class InterviewQuestion(BaseModel):
    question: str
    explanation: str 

class InterviewResponse(BaseModel):
    questions: list[InterviewQuestion]

class LiveInterviewRequest(BaseModel):
    target_role: str
    job_description: str  
    vault_data: str
    chat_history: str 
    user_audio_text: str
    elapsed_seconds: int 

class LiveInterviewResponse(BaseModel):
    ai_reply: str 
    is_concluded: bool 

class InterviewFeedbackRequest(BaseModel):
    target_role: str
    chat_history: str

class InterviewFeedbackResponse(BaseModel):
    hireability: str
    communication_feedback: str
    technical_feedback: str
    improvement_areas: list[str]

# ==========================================
# 3. AI ENDPOINTS
# ==========================================
@app.post("/v1/ai/parse-dump")
async def parse_brain_dump(req: BrainDumpRequest):
    # ✨ FIXED: Updated prompt to look for personal info
    prompt = f"""
    You are 'Rehan's Career Agent'. Your goal is to build a perfect resume through conversation.
    USER INPUT: "{req.transcript}"
    TASK:
    1. Extract Projects and Experience into the exact schema. 
    2. Write professional bullets using strong action verbs.
    3. Missing Fields: If start/end dates or company names are missing, list them in `missing_fields`.
    4. Skills: Suggest 3-5 technical skills based ONLY on the projects mentioned.
    5. Personal Info: If the user mentions their name, target role, or 'about me'/summary details, extract them into `first_name`, `last_name`, `target_role`, and `summary`.
    6. Reply (CRITICAL): Write a natural, human-like response (in the `reply` field). 
       - Tell the user what you saved. 
       - If dates are missing, politely ask the user to provide them. 
    Return ONLY JSON matching the schema.
    """
    try:
        client = get_ai_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeExtraction,
                temperature=0.7
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"CRASH: {str(e)}")
        return {"reply": "I'm here! Tell me about your recent projects or experience.", "first_name": "", "last_name": "", "target_role": "", "summary": "", "skills_suggested": [], "experience": [], "projects": [], "missing_fields": []}

@app.post("/v1/ai/cover-letter")
async def generate_cover_letter(req: CoverLetterRequest):
    prompt = f"Write a highly professional cover letter based on this JD:\n{req.job_description}\n\nUser data:\n{req.vault_data}"
    try:
        client = get_ai_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=CoverLetterResponse, temperature=0.7),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/analytics")
async def analyze_vault(req: AnalyticsRequest):
    prompt = f"Analyze this profile for '{req.target_role}':\n{req.vault_data}\nProvide 3 strengths and 3 missing skills."
    try:
        client = get_ai_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=AnalyticsResponse, temperature=0.7),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/generate-interview")
async def generate_interview(req: InterviewRequest):
    prompt = f"""
    You are an expert technical interviewer hiring for a '{req.target_role}' role.
    Based on this Job Description:
    {req.job_description}
    
    And the candidate's actual experience and projects:
    {req.vault_data}
    
    Generate exactly 4 highly specific interview questions. 
    - 2 Technical questions based on their projects/skills.
    - 2 Behavioral/Scenario questions based on the job description.
    For each, provide the 'question' and a brief 'explanation' of what a senior interviewer is actually looking for in the answer.
    """
    try:
        client = get_ai_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=InterviewResponse, 
                temperature=0.7
            ),
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        return json.loads(raw_text)
    except Exception as e:
        print(f"INTERVIEW CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/live-interview")
async def live_interview_turn(req: LiveInterviewRequest):
    prompt = f"""
    You are a senior technical interviewer conducting a live, spoken voice interview for a '{req.target_role}' position.
    
    Job Description for this specific role:
    {req.job_description}
    
    Candidate's Background:
    {req.vault_data}
    
    Past Conversation History:
    {req.chat_history}
    
    The Candidate just answered: "{req.user_audio_text}"
    
    YOUR INSTRUCTIONS:
    1. Act strictly as the interviewer. Do not break character.
    2. TIME LIMIT: This interview has a 5-minute (300 seconds) time limit. Currently, {req.elapsed_seconds} seconds have passed.
    3. IF the candidate says they want to end early (e.g., "that's it", "I'm done") OR if elapsed_seconds >= 300: 
       - You MUST conclude the interview gracefully. 
       - Thank the candidate for their time, tell them the team will be in touch.
       - EXPLICITLY set the 'is_concluded' JSON flag to true. 
       - Do NOT ask any more questions.
    4. IF the interview is NOT concluding:
       - Respond to what the candidate just said naturally.
       - Ask the NEXT interview question based heavily on the Job Description and their background.
       - Keep 'is_concluded' false.
    5. CRITICAL TTS CONSTRAINT: Keep it EXTREMELY CONCISE (1 to 3 short sentences maximum). Speak like a real human. Do not use markdown.
    """
    try:
        client = get_ai_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=LiveInterviewResponse, 
                temperature=0.7
            ),
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        return json.loads(raw_text)
    except Exception as e:
        print(f"LIVE INTERVIEW CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/interview-feedback")
async def generate_interview_feedback(req: InterviewFeedbackRequest):
    prompt = f"""
    You are an expert tech recruiter evaluating a candidate for a '{req.target_role}' role.
    Review the following complete interview transcript:
    
    {req.chat_history}
    
    Generate a brutally honest, constructive feedback report.
    - hireability: Give a brief assessment (e.g., "Strong Hire", "Needs Improvement", "Solid Candidate").
    - communication_feedback: Did they speak clearly? Did they ramble? Were they confident?
    - technical_feedback: Were their technical answers accurate and aligned with the role?
    - improvement_areas: List exactly 3 specific things they must improve before a real interview.
    """
    try:
        client = get_ai_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=InterviewFeedbackResponse, 
                temperature=0.7
            ),
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        return json.loads(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. PDF GENERATION LOGIC & ENDPOINTS
# ==========================================
def _wrap(text: str, font: str, size: int, max_width: float) -> list[str]:
    if not text:
        return []
    words = text.replace("\t", " ").split()
    lines, cur = [], []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if stringWidth(trial, font, size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _split_paragraphs(block: str) -> list[str]:
    if not block:
        return []
    raw = block.replace("\r\n", "\n").replace("\r", "\n")
    out, buf = [], []
    for line in raw.split("\n"):
        if line.strip() == "":
            if buf:
                out.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line.rstrip())
    if buf:
        out.append("\n".join(buf).strip())
    return [p for p in out if p.strip()]

def _is_bullet(line: str) -> bool:
    return line.strip().startswith(("•", "-", "*"))

def _clean_bullet(line: str) -> str:
    s = line.strip()
    return s[1:].strip() if s.startswith(("•", "-", "*")) else s

def _dedupe(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for item in seq:
        s = item.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out

def _tokenize_jd(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{1,}", text.lower())
    stop = {"the", "and", "with", "for", "you", "your", "that", "this", "from", "have", "has", "will", "are", "our", "job", "role", "team", "work", "year", "years", "plus"}
    return {t for t in tokens if len(t) > 2 and t not in stop}

def _prioritize_skills(skills: list[str], jd_text: str) -> list[str]:
    skills = _dedupe(skills)
    jd_tokens = _tokenize_jd(jd_text)
    return sorted(skills, key=lambda s: (0 if any(tok in s.lower() for tok in jd_tokens) else 1, s.lower()))

def _links_line(payload: ResumePdfRequest) -> str:
    parts = []
    if payload.linkedin.strip():
        parts.append(f"LinkedIn: {payload.linkedin.strip()}")
    if payload.github.strip():
        parts.append(f"GitHub: {payload.github.strip()}")
    if payload.portfolio.strip():
        parts.append(f"Portfolio: {payload.portfolio.strip()}")
    return " • ".join(parts)

def _image_reader_from_b64(raw_b64: str) -> ImageReader | None:
    raw = (raw_b64 or "").strip()
    if not raw:
        return None
    try:
        if "," in raw and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        raw = raw.replace("\n", "").replace("\r", "")
        data = base64.b64decode(raw, validate=False)
        return ImageReader(BytesIO(data))
    except Exception:
        return None

def _render_block_generic(c: canvas.Canvas, title: str, block: str, x: float, y: float, maxw: float, bottom: float, body_size: int, line_gap: int, draw_section, ensure_space) -> float:
    blk = block.strip()
    if not blk:
        return y
    y = draw_section(title, y)
    paras = _split_paragraphs(blk)
    for p in paras:
        for ln in p.split("\n"):
            if not ln.strip():
                continue
            if _is_bullet(ln):
                bullet = _clean_bullet(ln)
                wrapped = _wrap(bullet, "Helvetica", body_size, maxw - 14)
                if wrapped:
                    y = ensure_space(y, line_gap + 2)
                    c.setFont("Helvetica", body_size)
                    c.drawString(x, y, "•")
                    c.drawString(x + 12, y, wrapped[0])
                    y -= line_gap
                    for extra in wrapped[1:]:
                        y = ensure_space(y, line_gap + 2)
                        c.drawString(x + 12, y, extra)
                        y -= line_gap
            else:
                for wrapped in _wrap(ln.strip(), "Helvetica", body_size, maxw):
                    y = ensure_space(y, line_gap + 2)
                    c.setFont("Helvetica", body_size)
                    c.drawString(x, y, wrapped)
                    y -= line_gap
        y -= 6
    return y

def _build_ats_pdf(payload: ResumePdfRequest) -> bytes:
    top = 54
    left = 54
    right = 54
    bottom = 54
    body_size = 10
    section_size = 11
    line_gap = 12
    section_gap = 8

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    x = left
    maxw = width - left - right

    def reset_y() -> float:
        return height - top

    y = reset_y()

    def new_page() -> float:
        c.showPage()
        return reset_y()

    def ensure_space(cur_y: float, needed: float) -> float:
        if cur_y - needed < bottom:
            return new_page()
        return cur_y

    def draw_section(title: str, cur_y: float) -> float:
        cur_y = ensure_space(cur_y, 28) - section_gap
        c.setFont("Helvetica-Bold", section_size)
        c.drawString(x, cur_y, title.upper())
        cur_y -= 6
        c.setLineWidth(0.6)
        c.setStrokeGray(0.6)
        c.line(x, cur_y, x + maxw, cur_y)
        c.setStrokeGray(0)
        return cur_y - 10

    full_name = (payload.first_name + " " + payload.last_name).strip() or "Resume"
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, full_name)
    y -= 20

    if payload.target_role.strip():
        c.setFont("Helvetica", 11)
        c.drawString(x, y, payload.target_role.strip())
        y -= 16

    contact_parts = [p for p in [payload.email.strip(), payload.phone.strip(), payload.location.strip()] if p]
    contact = " • ".join(contact_parts)
    if contact:
        c.setFont("Helvetica", 9)
        c.setFillGray(0.15)
        c.drawString(x, y, contact)
        c.setFillGray(0)
        y -= 14

    links = _links_line(payload)
    if links:
        c.setFont("Helvetica", 9)
        c.setFillGray(0.15)
        for ln in _wrap(links, "Helvetica", 9, maxw):
            y = ensure_space(y, 12)
            c.drawString(x, y, ln)
            y -= 12
        c.setFillGray(0)

    if payload.summary.strip():
        y = draw_section("Summary", y)
        for ln in _wrap(payload.summary.strip(), "Helvetica", body_size, maxw):
            y = ensure_space(y, line_gap + 2)
            c.setFont("Helvetica", body_size)
            c.drawString(x, y, ln)
            y -= line_gap

    skills = _prioritize_skills(payload.skills, payload.jd_text)
    if skills:
        y = draw_section("Skills", y)
        for ln in _wrap(", ".join(skills), "Helvetica", body_size, maxw):
            y = ensure_space(y, line_gap + 2)
            c.setFont("Helvetica", body_size)
            c.drawString(x, y, ln)
            y -= line_gap

    sections = [
        ("Experience", payload.experience_text),
        ("Projects", payload.projects_text),
        ("Education", payload.education_text),
        ("Additional", payload.extras_text)
    ]
    for title, block in sections:
        y = _render_block_generic(c, title, block, x, y, maxw, bottom, body_size, line_gap, draw_section, ensure_space)

    c.save()
    return buf.getvalue()

def _build_modern_pdf(payload: ResumePdfRequest) -> bytes:
    page_w, page_h = LETTER
    margin_x = 42
    bottom = 42
    header_h = 96
    body_size = 10
    line_gap = 12
    title_color = colors.HexColor("#17324D")
    accent_color = colors.HexColor("#2F6B8F")
    soft_color = colors.HexColor("#EDF3F8")
    header_color = colors.HexColor("#163047")

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    img = _image_reader_from_b64(payload.profile_image_b64)

    def draw_header() -> float:
        c.setFillColor(header_color)
        c.rect(0, page_h - header_h, page_w, header_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        
        full_name = (payload.first_name + " " + payload.last_name).strip() or "Resume"
        role = payload.target_role.strip()
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(page_w / 2, page_h - 34, full_name)
        
        if role:
            c.setFont("Helvetica", 11)
            c.drawCentredString(page_w / 2, page_h - 52, role)
            
        contact_parts = [p for p in [payload.email.strip(), payload.phone.strip(), payload.location.strip()] if p]
        contact = " • ".join(contact_parts)
        if contact:
            c.setFont("Helvetica", 9)
            c.drawCentredString(page_w / 2, page_h - 68, contact)
            
        links = _links_line(payload)
        if links:
            c.setFont("Helvetica", 8)
            c.drawCentredString(page_w / 2, page_h - 80, links[:150])
            
        if img is not None:
            try:
                img_size = 56
                img_x = page_w - margin_x - img_size - 10
                img_y = page_h - header_h + 20
                c.saveState()
                path = c.beginPath()
                path.circle(img_x + img_size/2, img_y + img_size/2, img_size/2)
                c.clipPath(path, stroke=0)
                c.drawImage(img, img_x, img_y, width=img_size, height=img_size, preserveAspectRatio=True)
                c.restoreState()
            except Exception:
                pass
        return page_h - header_h - 18

    y = draw_header()
    maxw = page_w - (margin_x * 2)
    x = margin_x

    def new_page() -> float:
        c.showPage()
        return draw_header()

    def ensure_space(cur_y: float, needed: float) -> float:
        if cur_y - needed < bottom:
            return new_page()
        return cur_y

    def draw_section(title: str, cur_y: float) -> float:
        cur_y = ensure_space(cur_y - 14, 35)
        c.setFillColor(soft_color)
        c.roundRect(x, cur_y - 6, maxw, 20, 8, fill=1, stroke=0)
        c.setFillColor(accent_color)
        c.rect(x + 8, cur_y - 2, 4, 12, fill=1, stroke=0)
        c.setFillColor(title_color)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x + 18, cur_y + 1, title.upper())
        c.setFillColor(colors.black)
        return cur_y - 18

    if payload.summary.strip():
        y = draw_section("Summary", y)
        for ln in _wrap(payload.summary.strip(), "Helvetica", body_size, maxw):
            y = ensure_space(y, line_gap + 2)
            c.setFont("Helvetica", body_size)
            c.drawString(x, y, ln)
            y -= line_gap

    skills = _prioritize_skills(payload.skills, payload.jd_text)
    if skills:
        y = draw_section("Core Skills", y)
        for ln in _wrap(" • ".join(skills), "Helvetica", body_size, maxw):
            y = ensure_space(y, line_gap + 2)
            c.setFont("Helvetica", body_size)
            c.drawString(x, y, ln)
            y -= line_gap

    sections = [
        ("Experience", payload.experience_text),
        ("Projects", payload.projects_text),
        ("Education", payload.education_text),
        ("Additional", payload.extras_text)
    ]
    for title, block in sections:
        y = _render_block_generic(c, title, block, x, y, maxw, bottom, body_size, line_gap, draw_section, ensure_space)

    c.save()
    return buf.getvalue()

@app.post("/v1/resume/pdf")
def generate_resume_pdf(payload: ResumePdfRequest, x_app_key: str | None = Header(default=None)):
    tpl = (payload.template or "ats").strip().lower()
    is_modern = any(key in tpl for key in ["modern", "human", "graphic"])
    pdf_bytes = _build_modern_pdf(payload) if is_modern else _build_ats_pdf(payload)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": 'inline; filename="resume.pdf"'})