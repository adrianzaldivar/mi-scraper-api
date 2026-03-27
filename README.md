---
title: Mi Scraper API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Mi Scraper API

API REST con Playwright + GPT-4o-mini para scraping real de sitios web.

## Endpoints
- `GET /health` — health check
- `POST /run` — `{"company": "Tesla", "website": "https://tesla.com"}`
- `POST /score` — `{"description": "...", "evaluation_goal": "...", "evaluation_type": "score"}`
