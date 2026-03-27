from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple
import os, json, asyncio, logging

from openai import AsyncOpenAI
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scraper API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CONTENT_LENGTH = 15000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 5


class BrowserManager:
    _instance: Optional["BrowserManager"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._initialized = False

    @classmethod
    async def get_instance(cls) -> "BrowserManager":
        async with cls._lock:
            if cls._instance is None:
                instance = cls()
                await instance._init()
                cls._instance = instance
            return cls._instance

    async def _init(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-dev-shm-usage","--disable-gpu",
                  "--disable-extensions","--disable-background-networking"],
        )
        self._initialized = True
        logger.info("Chromium browser ready.")

    async def get_page_content(self, url: str) -> Tuple[str, str]:
        if not self._initialized or self._browser is None:
            raise RuntimeError("Browser not initialized.")
        context = await self._browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        try:
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            raw_html = await page.content()
        finally:
            await page.close()
            await context.close()
        return _clean_html(raw_html), raw_html

    async def scroll_and_get_content(self, url: str) -> str:
        if not self._initialized or self._browser is None:
            raise RuntimeError("Browser not initialized.")
        context = await self._browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        try:
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
                await asyncio.sleep(0.4)
            raw_html = await page.content()
        finally:
            await page.close()
            await context.close()
        return _clean_html(raw_html)


def _clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["script","style","nav","footer","noscript","iframe","svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:MAX_CONTENT_LENGTH]


SYSTEM_PROMPT = (
    "You are a web research agent. Visit URLs and extract the requested information. "
    "You MUST start by calling navigate_to_url to visit the website. "
    "NEVER answer from memory — always use the tools to navigate first. "
    "When you have enough information, respond with the final answer without calling more tools."
)

TOOLS: List[Dict[str, Any]] = [
    {"type":"function","function":{"name":"navigate_to_url","description":"Navigate to a URL using a real browser and return the page title and full visible text content.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
    {"type":"function","function":{"name":"scroll_and_get_more_content","description":"Scroll down the current page to load lazy content.","parameters":{"type":"object","properties":{},"required":[]}}},
]


class ScraperAgent:
    def __init__(self, prompt: str, url: str) -> None:
        self.prompt = prompt
        self.url = url
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self._current_page_content: str = ""
        self._current_url: str = url
        self._tokens_used: int = 0

    async def run(self) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {self.prompt}\nURL: {self.url}"},
        ]
        final_result: Any = None

        for iteration in range(MAX_ITERATIONS):
            tool_choice: Any = (
                {"type": "function", "function": {"name": "navigate_to_url"}}
                if iteration == 0 else "auto"
            )
            response = await self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice=tool_choice,
            )
            if response.usage:
                self._tokens_used += response.usage.total_tokens

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=False))

            if not msg.tool_calls:
                final_result = msg.content
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                if fn_name == "navigate_to_url":
                    result = await self._navigate(fn_args.get("url", self.url))
                elif fn_name == "scroll_and_get_more_content":
                    result = await self._scroll()
                else:
                    result = f"Unknown tool: {fn_name}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            last = messages[-1]
            final_result = last.get("content", "Max iterations reached.") if isinstance(last, dict) else "Max iterations reached."

        return {"result": final_result, "tokens_used": self._tokens_used}

    async def _navigate(self, url: str) -> str:
        try:
            browser = await BrowserManager.get_instance()
            clean_text, raw_html = await browser.get_page_content(url)
            self._current_page_content = clean_text
            self._current_url = url
            soup = BeautifulSoup(raw_html, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
            return f"Title: {title}\n\nContent:\n{clean_text}"
        except Exception as exc:
            return f"Error navigating to {url}: {exc}"

    async def _scroll(self) -> str:
        if not self._current_url:
            return "No URL loaded."
        try:
            browser = await BrowserManager.get_instance()
            content = await browser.scroll_and_get_content(self._current_url)
            self._current_page_content = content
            return f"Scrolled content:\n{content}"
        except Exception as exc:
            return f"Error scrolling: {exc}"


class RunRequest(BaseModel):
    company: str
    website: str

class ScoreRequest(BaseModel):
    description: str
    evaluation_goal: str
    evaluation_type: str = "score"
    categories: Optional[List[str]] = None
    score_threshold: int = 60


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
async def run(req: RunRequest):
    prompt = (
        f"Research on the web what this company does and give me a brief description "
        f"of their activity in english. This is the company: {req.company}, "
        f"this is their website: {req.website}"
    )
    agent = ScraperAgent(prompt=prompt, url=req.website)
    result = await agent.run()
    return {"company": req.company, "website": req.website, "result": result["result"], "tokens_used": result["tokens_used"]}

@app.post("/score")
async def score(req: ScoreRequest):
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    if req.evaluation_type == "score":
        format_instruction = "The output format must be ONLY a single integer from 1 to 100. No explanation, no text, just the number."
        threshold_instruction = (
            f"CRITICAL THRESHOLD RULE: Score {req.score_threshold}+ = company WILL be contacted. "
            f"Score below {req.score_threshold} = will NOT be contacted. Be conservative. "
            f"Reserve 80-100 for exceptional fits."
        )
    else:
        cats = ", ".join(req.categories or [])
        format_instruction = f"The output format must be ONLY one of: {cats}. No explanation, just the label."
        threshold_instruction = ""

    meta_prompt = (
        "You are an expert B2B lead scoring prompt engineer.\n"
        "Write a concise system prompt to evaluate companies based on a short description.\n"
        f'Evaluation goal: "{req.evaluation_goal}"\n'
        f"{threshold_instruction}\n"
        f"Output format: {format_instruction}\n"
        "The prompt must be under 350 words, in English, output ONLY the final value with no explanation.\n"
        "Return ONLY the system prompt text."
    )
    meta_resp = await client.chat.completions.create(
        model=MODEL, messages=[{"role":"user","content":meta_prompt}], temperature=0.3, max_tokens=600,
    )
    scoring_prompt = meta_resp.choices[0].message.content.strip()
    score_resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":scoring_prompt},{"role":"user","content":f"Company description:\n{req.description}"}],
        temperature=0, max_tokens=10,
    )
    raw = score_resp.choices[0].message.content.strip()
    tokens = meta_resp.usage.total_tokens + score_resp.usage.total_tokens
    output: Dict[str, Any] = {"result": raw, "tokens_used": tokens}
    if req.evaluation_type == "score":
        try:
            s = int(raw)
            output["score"] = s
            output["decision"] = "INCLUDED" if s >= req.score_threshold else "EXCLUDED"
        except ValueError:
            output["decision"] = "ERROR"
    return output
