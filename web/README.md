# Restaurant Web

Frontend React/Vite lives here. Backend da tach rieng hoan toan sang thu muc [../backend](../backend).

## Run frontend

```bash
npm install
npm run dev
```

## Notes

- Frontend goi API qua duong dan `/rank` va `/health`, Vite se proxy local dev sang `http://localhost:8000`.
- Neu deploy frontend va backend o 2 domain khac nhau, set `VITE_API_BASE_URL` luc build frontend.
- Backend co the doc env tu `BACKEND_ENV_FILE`, root `.env`, hoac `backend/.env`.
- Model artifact co the nam ngoai repo neu set `RANKER_ARTIFACT_PATH`.
