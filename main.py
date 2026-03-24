from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class ScrapeRequest(BaseModel):
    company: str
    website: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(req: ScrapeRequest):
    prompt = (
        f"Research on the web what this company does and give me a brief description "
        f"of their activity in english. This is the company: {req.company}, "
        f"this is their website: {req.website}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return {
        "company": req.company,
        "website": req.website,
        "result": response.choices[0].message.content.strip(),
        "tokens_used": response.usage.total_tokens
    }
