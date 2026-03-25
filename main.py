import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# NUCLEAR FIX: Completely renamed the variable so Render cannot use the old cached key
api_key = os.getenv("GEMINI_LIVE_KEY", "")
client = genai.Client(api_key=api_key)

# ==========================================
# 1. PARSE DUMP (CHATBOT) MODELS
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

class ResumeExtraction(BaseModel):
    reply: str
    summary: str
    skills_suggested: list[str]
    experience: list[Experience]
    projects: list[Project]
    missing_fields: list[str]

# ==========================================
# 2. NEW: COVER LETTER & ANALYTICS MODELS
# ==========================================
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

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/v1/ai/parse-dump")
async def parse_brain_dump(req: BrainDumpRequest):
    prompt = f"""
    You are 'Rehan's Career Agent'. Your goal is to build a perfect resume through conversation.
    
    USER INPUT: "{req.transcript}"

    TASK:
    1. Extract Projects and Experience into the exact schema. 
    2. Write professional bullets using strong action verbs.
    3. Missing Fields: If start/end dates or company names are missing, list them in `missing_fields`.
    4. Skills: Suggest 3-5 technical skills based ONLY on the projects mentioned.
    5. Reply (CRITICAL): Write a natural, human-like response (in the `reply` field). 
       - Tell the user what you saved. 
       - If dates are missing, politely ask the user to provide them. 
       - If they just gave a project, ask if they want to add skills, education, or another project.
    
    Return ONLY JSON matching the schema.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeExtraction,
                temperature=0.7
            ),
        )
        
        parsed_data = json.loads(response.text)
        return parsed_data
        
    except Exception as e:
        print(f"CRASH: {str(e)}")
        # Safe fallback so the Android app doesn't crash if the AI fails
        return {
            "reply": "I'm here! Tell me about your recent projects or experience.", 
            "summary": "", "skills_suggested": [], "experience": [], "projects": [], "missing_fields": []
        }

@app.post("/v1/ai/cover-letter")
async def generate_cover_letter(req: CoverLetterRequest):
    prompt = f"""
    Write a highly professional, engaging cover letter based on this Job Description:
    {req.job_description}
    
    And this user data from their Master Vault:
    {req.vault_data}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CoverLetterResponse,
                temperature=0.7
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/analytics")
async def analyze_vault(req: AnalyticsRequest):
    prompt = f"""
    Analyze this user's profile for a '{req.target_role}' role:
    {req.vault_data}
    
    Provide exactly 3 key strengths they have, and 3 missing skills/gaps they should learn or add.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalyticsResponse,
                temperature=0.7
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))