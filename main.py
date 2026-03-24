from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class ScrapeRequest(BaseModel):
    url: str
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(req: ScrapeRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": req.prompt},
            {"role": "user", "content": f"Empresa: {req.url}"}
        ],
        max_tokens=100
    )
    return {
        "url": req.url,
        "result": response.choices[0].message.content.strip(),
        "tokens_used": response.usage.total_tokens
    }
