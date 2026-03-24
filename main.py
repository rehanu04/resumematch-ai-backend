import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Make sure your API key is either set in the terminal or pasted here
api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key="AIzaSyDY9SrbJpioYSqEGFtGo5DT09KVG33-vIc")

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

# The upgraded schema with Conversational Reply and Skill Suggestions
class ResumeExtraction(BaseModel):
    reply: str
    summary: str
    skills_suggested: list[str]
    experience: list[Experience]
    projects: list[Project]
    missing_fields: list[str]

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
        raise HTTPException(status_code=500, detail=str(e))