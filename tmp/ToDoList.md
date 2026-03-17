# 취업나침반 (Job Compass) — Master TODO List

> **Status**: 🟡 Planning Phase
> **Last Updated**: 2026-03-13
> **Progress**: 0/48 tasks complete

---

## Phase 1: Data Collection & Database (Week 1-4)

### 1.1 Data Source Setup
- [ ] P1-001: Register KOSIS API key and test connection
- [ ] P1-002: Register 한국은행 ECOS API key (CPI/inflation data)
- [ ] P1-003: Register 워크넷 (work.go.kr) API access
- [ ] P1-004: Set up 대학알리미 scraper (academyinfo.go.kr)
- [ ] P1-005: Set up KEDI GOMS dataset download pipeline
- [ ] P1-006: Evaluate 사람인/잡코리아 scraping feasibility (legal check)

### 1.2 Data Pipeline
- [ ] P1-007: Design PostgreSQL schema (majors, regions, salaries, employment)
- [ ] P1-008: Build ETL pipeline — raw ingestion scripts
- [ ] P1-009: Build data cleaning & normalization module
- [ ] P1-010: Create CPI-adjusted real salary calculation module
- [ ] P1-011: Build major name standardization mapping (전공명 통일)
- [ ] P1-012: Build region code standardization (시도/시군구 통일)

### 1.3 Validation
- [ ] P1-013: Validate 10-year data coverage (2015-2025) per source
- [ ] P1-014: Cross-validate employment rates across KOSIS vs 대학알리미
- [ ] P1-015: Document data gaps and limitations

---

## Phase 2: Analytics & Visualization (Week 5-8)

### 2.1 Core Analytics Engine
- [ ] P2-001: Employment rate by major (전공별 취업률) — 10yr trend
- [ ] P2-002: Employment rate by region (지역별) — 10yr trend
- [ ] P2-003: Initial salary vs CPI inflation (실질임금) — 10yr trend
- [ ] P2-004: Employment type breakdown (정규직/비정규직/프리랜서)
- [ ] P2-005: Time-to-employment analysis (졸업후 취업소요기간)
- [ ] P2-006: Job-major mismatch rate (전공불일치율)
- [ ] P2-007: University tier effect analysis (대학서열 영향력)
- [ ] P2-008: Industry sector shift analysis (산업별 채용변화)
- [ ] P2-009: Gender gap trend analysis (성별격차)
- [ ] P2-010: Graduate school escape rate (대학원 진학률)
- [ ] P2-011: NEET rate by major (전공별 니트족)

### 2.2 Frontend Dashboard
- [ ] P2-012: Scaffold Next.js project with dark theme
- [ ] P2-013: Build Dashboard page — national overview cards
- [ ] P2-014: Build Major Explorer page — major deep-dive with charts
- [ ] P2-015: Build Regional Map page — interactive Korea heatmap
- [ ] P2-016: Build Salary Calculator page — real wage calculator
- [ ] P2-017: Build Compare Tool page — side-by-side comparison
- [ ] P2-018: Implement responsive mobile layout

### 2.3 Backend API
- [ ] P2-019: Set up FastAPI project structure
- [ ] P2-020: Build REST endpoints for all 10 analytics dimensions
- [ ] P2-021: Implement query caching layer
- [ ] P2-022: Add GZip compression middleware

---

## Phase 3: AI Impact Analysis & Forecasting (Week 9-12)

### 3.1 AI Impact Scoring
- [ ] P3-001: Build AI Exposure Index per major (O*NET Korea adaptation)
- [ ] P3-002: Classify majors into AI-risk tiers (High/Medium/Low)
- [ ] P3-003: Analyze pre/post 2023 employment shift by AI-risk tier
- [ ] P3-004: NLP pipeline — extract skill demands from 워크넷 job postings
- [ ] P3-005: Detect emerging vs declining skills over time

### 3.2 Hypothesis Testing
- [ ] P3-006: Test H1 — 경영/행정 employment decline post-2023
- [ ] P3-007: Test H2 — CS/AI salary premium acceleration
- [ ] P3-008: Test H3 — Humanities U-shaped recovery
- [ ] P3-009: Test H4 — Regional gap widening (Seoul/Pangyo concentration)
- [ ] P3-010: Test H5 — Job-major mismatch increase
- [ ] P3-011: Test H6 — Mid-tier university time-to-employment lag

### 3.3 Forecasting Models
- [ ] P3-012: ARIMA/Prophet — employment rate time-series forecast
- [ ] P3-013: XGBoost — multi-factor employment predictor
- [ ] P3-014: Build confidence interval visualization
- [ ] P3-015: Build AI Impact Tracker page in frontend

---

## Phase 4: Polish & Launch (Week 13-16)

### 4.1 Features
- [ ] P4-001: Add user bookmark / save comparison feature
- [ ] P4-002: Add data export (CSV/PDF) functionality
- [ ] P4-003: Add Korean/English language toggle
- [ ] P4-004: SEO optimization for key search terms

### 4.2 Quality & Deploy
- [ ] P4-005: Performance audit — Lighthouse score ≥ 90
- [ ] P4-006: Write unit tests for analytics engine (≥80% coverage)
- [ ] P4-007: Set up CI/CD pipeline
- [ ] P4-008: Deploy to production (Vercel + cloud DB)
- [ ] P4-009: Write project documentation & README

---

## Tracking Legend

| Symbol | Meaning |
|--------|---------|
| `- [ ]` | Not started |
| `- [x]` | Completed |
| `P1-XXX` | Phase 1 task ID |
| `P2-XXX` | Phase 2 task ID |
| `P3-XXX` | Phase 3 task ID |
| `P4-XXX` | Phase 4 task ID |

---

## AI Update Instructions

To mark a task complete, replace `- [ ]` with `- [x]` for the matching task ID.
To update progress counter, recalculate completed vs total in the header.

Example:
```
Before: - [ ] P1-001: Register KOSIS API key and test connection
After:  - [x] P1-001: Register KOSIS API key and test connection
```
