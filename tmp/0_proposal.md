# Korean College Graduate Employment Analytics Platform

## Proposal: "취업나침반" (Job Compass)

---

## Core Analysis Dimensions

### Primary 4 Factors
1. **Major/전공별**: Employment rate, time-to-hire, job-major alignment rate
2. **Regional/지역별**: Seoul vs metro vs rural, regional brain drain patterns
3. **Initial Salary vs Inflation**: Real wage purchasing power over time, by major & region
4. **Employment Type** (정규직 vs 비정규직 vs 프리랜서): Contract quality is degrading — raw employment rate hides this

### Recommended Additional Factors

| # | Factor | Why It Matters |
|---|--------|---------------|
| 5 | **Time-to-Employment** (졸업후 취업소요기간) | Growing gap between graduation and first job |
| 6 | **Job-Major Mismatch Rate** (전공불일치율) | Are degrees becoming irrelevant? Critical for AI thesis |
| 7 | **University Tier Effect** (대학서열 영향력) | Is the SKY premium shrinking or growing? |
| 8 | **Industry Sector Shift** (산업별 채용변화) | Which industries are hiring/dying — direct AI impact signal |
| 9 | **Certification & Skill Premium** (자격증/스킬 프리미엄) | Do certs/bootcamps outperform degrees? |
| 10 | **Gender Gap Trend** (성별격차 추이) | Wage & employment gap trajectory |
| 11 | **Graduate School Escape Rate** (대학원 진학률) | Rising grad school = bad job market signal |
| 12 | **NEET Rate by Major** (전공별 니트족 비율) | Not employed, not in education, not in training |

---

## AI Impact Thesis Framework

```
Phase 1 (2015-2019): Pre-AI baseline
Phase 2 (2020-2022): COVID disruption + remote work shift
Phase 3 (2023-2025): AI adoption begins — early displacement signals
Phase 4 (2026-2030): Forecasting zone — model predictions
```

### Key AI Hypotheses to Test
- **H1**: Administrative/clerical majors (경영, 행정) show declining employment rates post-2023
- **H2**: CS/AI majors show salary premium acceleration
- **H3**: Creative + humanities majors show U-shaped recovery (AI can't fully replace)
- **H4**: Regional gap widens as AI jobs concentrate in Seoul/Pangyo
- **H5**: Job-major mismatch rate increases across all fields
- **H6**: Time-to-employment increases for mid-tier universities faster than top-tier

---

## Data Sources (Korean Public Data)

| Source | Data | URL/API |
|--------|------|---------|
| **KOSIS** (통계청) | Employment by major, region, salary | kosis.kr |
| **KEDI** (한국교육개발원) | 대졸자직업이동경로조사 (GOMS) | kedi.re.kr |
| **워크넷** | Job postings, demand trends | work.go.kr API |
| **대학알리미** | University-level employment stats | academyinfo.go.kr |
| **한국은행** | CPI/inflation data | ecos.bok.or.kr API |
| **KEIS** (고용정보원) | 대졸자 취업통계 | keis.or.kr |
| **사람인/잡코리아** | Real-time salary & posting data (scraping) | — |

---

## Platform Architecture (High-Level)

```
┌─────────────────────────────────────────────┐
│              Frontend (Next.js)              │
│  Dashboard │ Explorer │ Forecaster │ Compare │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│              Backend (FastAPI)               │
│  REST API │ Analytics Engine │ ML Pipeline   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│           Data Layer (PostgreSQL)            │
│  Raw Data │ Processed │ Forecasts │ Cache    │
└─────────────────────────────────────────────┘
```

### Key Pages
1. **Dashboard**: National overview with key metrics & trend cards
2. **Major Explorer**: Deep dive by major — employment, salary, AI risk score
3. **Regional Map**: Interactive Korea map with regional employment heatmap
4. **AI Impact Tracker**: Before/after AI adoption metrics by sector
5. **Salary Calculator**: Real salary vs inflation, by major + region + year
6. **Forecaster**: ML-based 5-year projections with confidence intervals
7. **Compare Tool**: Major vs major, region vs region, university tier vs tier

---

## Forecasting Models

| Model | Purpose |
|-------|---------|
| **ARIMA/Prophet** | Time-series trend extrapolation |
| **XGBoost** | Multi-factor employment prediction |
| **NLP on job postings** | Skill demand shift detection |
| **AI Exposure Index** | Per-major AI displacement risk (based on O*NET methodology adapted for Korea) |

---

## Monetization / Sustainability

- **Free tier**: Basic dashboard, national trends
- **Premium**: University-specific reports, forecasting, API access
- **B2B**: Universities buy reports for curriculum planning
- **Government**: Policy research partnerships (KEDI, 고용노동부)

---

## Phased Roadmap

| Phase | Scope | Timeline |
|-------|-------|----------|
| **Phase 1** | Data collection + cleaning + basic dashboard (3 core metrics) | 4 weeks |
| **Phase 2** | Full 10-factor analysis + interactive visualizations | 4 weeks |
| **Phase 3** | AI impact analysis + forecasting models | 4 weeks |
| **Phase 4** | Compare tools, premium features, mobile optimization | 4 weeks |

---

## Next Step Options

1. **Start with data** — Build scrapers/API clients for KOSIS, KEDI, 대학알리미 and structure the database
2. **Start with frontend** — Scaffold the Next.js dashboard with mock data to nail the UX first
3. **Start with analysis** — Build the Python analytics pipeline to validate hypotheses with real data
4. **Deep dive on one factor** — Pick one dimension (e.g., AI impact scoring) and prototype it end-to-end
