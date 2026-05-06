# Vercel Frontend Setup

This repo now has two frontends:

- Streamlit app: `app.py` (existing)
- Vercel frontend: `frontend/*` + `api/*` (new)

## Local dev

1. Keep using Streamlit normally:

```powershell
streamlit run app.py
```

2. For Vercel frontend preview (if Vercel CLI is installed):

```powershell
vercel dev
```

Open `http://localhost:3000`.

## Deploy to Vercel

1. Import this repo into Vercel.
2. Vercel will read `vercel.json`.
3. Static UI is served from `frontend/`.
4. Python endpoints are:
   - `/api/materials`
   - `/api/analyze?material=Zinc&n_avrami=4&n_virtual=200`

## Notes

- The API reuses `main.py` logic (`train_models`, `ttt_curve`, `ml_predict`).
- Models are cached per serverless instance and per `(n_avrami, n_virtual)` pair.
- First request for a new model setting can take longer due to training.
