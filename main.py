from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
import requests
from dotenv import load_dotenv

# Supabase ve LangChain (v0.1.20 Uyumlu)
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
# DÜZELTME: Eski ama çalışan import adresine döndük
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
# DÜZELTME: Chains modülü v0.1.20 sürümünde buradadır
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Ortam Değişkenlerini Yükle
load_dotenv()

app = FastAPI()

# Ayarlar
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EVOLUTION_API_URL = os.environ.get("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.environ.get("EVOLUTION_API_KEY")

# Supabase İstemcisi
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Yapay Zeka Modelleri (CPU Dostu FastEmbed + Groq)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# Vektör Deposu Bağlantısı
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="document_chunks",
    query_name="match_documents"
)

# --- BASİT ADMİN PANELİ (HTML) ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Bot Yönetim Paneli</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container mt-5">
    <div class="card shadow">
        <div class="card-header bg-primary text-white">
            <h3>🤖 WhatsApp Bot Yönetim Paneli</h3>
        </div>
        <div class="card-body">
            <div class="mb-4">
                <h5>📁 Yeni Bilgi Yükle (PDF)</h5>
                <form action="/upload" method="post" enctype="multipart/form-data" class="d-flex gap-2">
                    <input type="file" name="file" class="form-control" accept=".pdf" required>
                    <button type="submit" class="btn btn-success">Yükle ve Öğret</button>
                </form>
            </div>
            <hr>
            <div class="mb-4">
                <h5>📞 İzinli Kişiler (Whitelist)</h5>
                <form action="/add-phone" method="post" class="d-flex gap-2 mb-3">
                    <input type="text" name="phone" placeholder="90555..." class="form-control" required>
                    <input type="text" name="name" placeholder="Ad Soyad" class="form-control">
                    <button type="submit" class="btn btn-primary">Ekle</button>
                </form>
                <ul class="list-group">
                    {% for p in whitelist %}
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        {{ p.name }} ({{ p.phone_number }})
                        <span class="badge bg-secondary">{{ p.trigger_word }}</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    try:
        response = supabase.table("whitelist").select("*").execute()
        whitelist = response.data
    except:
        whitelist = []
    from jinja2 import Template
    t = Template(html_template)
    return t.render(whitelist=whitelist)

@app.post("/upload")
async def upload_file(file: bytes =  Depends(lambda: None)): 
    # Demo modunda
    return {"status": "Dosya yükleme simülasyonu başarılı."}

@app.post("/add-phone")
async def add_phone(phone: str = Form(...), name: str = Form(...)):
    supabase.table("whitelist").insert({"phone_number": phone, "name": name}).execute()
    return {"status": "Kişi eklendi", "phone": phone}

# --- WEBHOOK (WHATSAPP BAĞLANTISI) ---
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        data = await request.json()
        print("Gelen Veri:", data)
        
        event_type = data.get("event")
        if event_type != "messages.upsert":
            return {"status": "ignored"}
            
        message_data = data.get("data", {})
        sender = message_data.get("key", {}).get("remoteJid", "").split("@")[0]
        text_body = message_data.get("message", {}).get("conversation") or message_data.get("message", {}).get("extendedTextMessage", {}).get("text")
        
        if not text_body:
            return {"status": "no text"}

        # 1. Whitelist Kontrolü
        user_check = supabase.table("whitelist").select("*").eq("phone_number", sender).execute()
        if not user_check.data:
            print(f"Yetkisiz numara: {sender}")
            return {"status": "unauthorized"}
            
        user_info = user_check.data[0]
        trigger = user_info.get("trigger_word", "@siri")
        
        # 2. Tetikleyici Kelime Kontrolü
        if not text_body.strip().lower().startswith(trigger.lower()):
            print("Tetikleyici kelime yok.")
            return {"status": "no trigger"}
            
        query = text_body[len(trigger):].strip()
        
        # 3. RAG: Cevap Üret
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        ai_response = qa_chain.run(query)
        
        # 4. Yanıtı Gönder
        send_url = f"{EVOLUTION_API_URL}/message/sendText/{data.get('instance')}"
        headers = {"apikey": EVOLUTION_API_KEY}
        payload = {"number": sender, "text": ai_response}
        requests.post(send_url, json=payload, headers=headers)
        
        return {"status": "sent", "response": ai_response}

    except Exception as e:
        print(f"Hata: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
