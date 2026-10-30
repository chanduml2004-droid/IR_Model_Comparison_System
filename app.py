import json, math, os, re
from collections import Counter, defaultdict
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "documents.json")

def load_documents():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)

def save_documents(docs):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

STOP_WORDS = set("a an the and or but is are was were to of in on for with from by as at be this that it".split())

def boolean_search(query, docs):
    # Supports AND, OR, NOT and parentheses-free expressions.
    raw = query.strip()
    if not raw:
        return []
    tokens = re.findall(r"\bAND\b|\bOR\b|\bNOT\b|\(|\)|[a-zA-Z0-9_]+", raw, flags=re.I)
    if not tokens:
        return []
    # Convert each document to a set and evaluate a safe Boolean expression.
    universe = {d["id"] for d in docs}
    term_sets = {t: {d["id"] for d in docs if t in set(tokenize(d["title"]+" "+d["content"]))}
                 for t in set(x.lower() for x in tokens if x.upper() not in {"AND","OR","NOT"})}
    # Shunting-yard to postfix; implicit AND between adjacent terms.
    expanded=[]
    for i,t in enumerate(tokens):
        if i and (tokens[i-1] not in ("AND","OR","NOT","(") and t not in ("AND","OR",")")):
            expanded.append("AND")
        expanded.append(t)
    prec={"OR":1,"AND":2,"NOT":3}
    output=[]; ops=[]
    for t in expanded:
        u=t.upper()
        if u in ("AND","OR","NOT"):
            while ops and ops[-1]!="(" and prec.get(ops[-1],0)>=prec[u]:
                output.append(ops.pop())
            ops.append(u)
        elif t=="(":
            ops.append(t)
        elif t==")":
            while ops and ops[-1]!="(": output.append(ops.pop())
            if ops: ops.pop()
        else: output.append(t.lower())
    while ops: output.append(ops.pop())
    stack=[]
    try:
        for t in output:
            if t=="NOT":
                stack.append(universe-stack.pop())
            elif t in ("AND","OR"):
                b=stack.pop(); a=stack.pop()
                stack.append(a & b if t=="AND" else a | b)
            else:
                stack.append(term_sets.get(t,set()))
        ids=stack[-1] if len(stack)==1 else set()
    except (IndexError, KeyError):
        ids=set()
    return [{"document":d,"score":1.0,"model":"Boolean"} for d in docs if d["id"] in ids]

def tfidf_search(query, docs):
    qterms=[t for t in tokenize(query) if t not in STOP_WORDS]
    if not qterms: return []
    corpus=[tokenize(d["title"]+" "+d["content"]) for d in docs]
    n=len(docs); df=Counter()
    for terms in corpus: df.update(set(terms))
    idf={term: math.log((n+1)/(df[term]+1))+1 for term in df}
    def vector(terms):
        counts=Counter(t for t in terms if t not in STOP_WORDS)
        return {t:(1+math.log(c))*idf.get(t,1) for t,c in counts.items()}
    qv=vector(qterms)
    def cosine(a,b):
        dot=sum(v*b.get(k,0) for k,v in a.items())
        na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0
    results=[]
    for d,terms in zip(docs,corpus):
        score=cosine(qv,vector(terms))
        if score>0: results.append({"document":d,"score":round(score,4),"model":"TF-IDF / Cosine"})
    return sorted(results,key=lambda x:x["score"],reverse=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/documents", methods=["GET","POST"])
def documents_api():
    docs=load_documents()
    if request.method=="GET": return jsonify(docs)
    data=request.get_json(silent=True) or {}
    title=(data.get("title") or "").strip()
    content=(data.get("content") or "").strip()
    if not title or not content: return jsonify({"error":"Both title and content are required."}),400
    new_id=max([d["id"] for d in docs],default=0)+1
    doc={"id":new_id,"title":title,"content":content}
    docs.append(doc); save_documents(docs)
    return jsonify(doc),201

@app.route("/api/search", methods=["POST"])
def search_api():
    data=request.get_json(silent=True) or {}
    query=(data.get("query") or "").strip()
    if not query: return jsonify({"error":"Enter a search query."}),400
    docs=load_documents()
    return jsonify({"query":query,"boolean":boolean_search(query,docs),
                    "tfidf":tfidf_search(query,docs),"total_documents":len(docs)})

@app.route("/health")
def health(): return jsonify({"status":"ok"})

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)
