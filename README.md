# IR Model Comparison System

A Flask web application that applies **two information retrieval models to the same document collection** and displays their results side by side.

## Models
1. **Boolean Retrieval** — exact term matching with `AND`, `OR`, and `NOT`.
2. **TF-IDF + Cosine Similarity** — weighted vector retrieval with ranked results and similarity scores.

## Features
- Search both models with one query.
- Compare Boolean matches against ranked TF-IDF results.
- Document Library showing the shared collection.
- Add documents through the website; new documents are immediately searchable by both models.
- JSON-backed storage in `data/documents.json`.
- Responsive interface and health endpoint at `/health`.

## Run locally
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy on Render
1. Push this project to a GitHub repository.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Use these settings:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
4. Deploy. The app binds to `0.0.0.0` and reads Render's `PORT` environment variable.

**Persistence note:** the starter app saves added documents to the JSON file. On Render, local filesystem changes may be lost when the service is redeployed or its instance is replaced unless you configure persistent storage or use a database.

## Project structure
```
ir_model_comparison_system/
├── app.py
├── requirements.txt
├── Procfile
├── README.md
├── data/
│   └── documents.json
├── templates/
│   └── index.html
└── static/
    ├── app.js
    └── style.css
```

## Example queries
- `information AND retrieval` — Boolean AND
- `machine OR database` — Boolean OR
- `retrieval NOT boolean` — Boolean NOT
- `machine learning` — TF-IDF ranking (Boolean parser treats adjacent terms as AND)

This is an educational comparison project. The two models intentionally have different retrieval behavior: Boolean retrieval returns matching documents without a relevance ranking, while TF-IDF orders matches by cosine similarity.
