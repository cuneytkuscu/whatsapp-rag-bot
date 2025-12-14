import os
import logging
import shutil
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyCookie
from pydantic import BaseModel

# --- AI & Database Libraries ---
from supabase import create_client, Client
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import requests

# --- Yapılandırma ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG WhatsApp Bot", version="1.0")

# Çevresel Değişkenler
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EVOLUTION_API_URL = os.environ.get("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.environ.get("EVOLUTION_API_KEY")
EVOLUTION_INSTANCE = os.environ.get("EVOLUTION_INSTANCE", "SiriBot")

# Supabase İstemcisi
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Embedding Modeli (Ücretsiz ve Docker dostu)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vektör Deposu Bağlantısı
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

# --- HTML Şablonları (Tek dosya içinde) ---
HTML_LOGIN = """
<!DOCTYPE html>
<html>
<head>
    <title>Admin Giriş</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 flex items-center justify-center h-screen">
    <div class="bg-gray-800 p-8 rounded-lg shadow-lg w-96">
        <h2 class="text-2xl text-white mb-6 text-center">Bot Admin Paneli</h2>
        <form action="/admin/login" method="post">
            <div class="mb-4">
                <label class="block text-gray-300 text-sm font-bold mb-2">Kullanıcı Adı</label>
                <input name="username" type="text" class="w-full p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-blue-500">
            </div>
            <div class="mb-6">
                <label class="block text-gray-300 text-sm font-bold mb-2">Şifre</label>
                <input name="password" type="password" class="w-full p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-blue-500">
            </div>
            <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200">Giriş Yap</button>
        </form>
    </div>
</body>
</html>
"""

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-200 min-h-screen">
    <nav class="bg-gray-800 p-4 shadow-md">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-xl font-bold text-blue-400">RAG Bot Yönetimi</h1>
            <a href="/admin/logout" class="text-red-400 hover:text-red-300">Çıkış</a>
        </div>
    </nav>
    
    <div class="container mx-auto p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
        <!-- Bilgi Yükleme Alanı -->
        <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 class="text-xl font-bold mb-4 text-white border-b border-gray-700 pb-2">📂 Bilgi Yükle (RAG)</h2>
            <p class="text-sm text-gray-400 mb-4">PDF veya TXT dosyası yükleyin. İçerik vektörlere dönüştürülüp veritabanına kaydedilecektir.</p>
            
            <form action="/admin/upload" method="post" enctype="multipart/form-data" class="space-y-4">
                <div class="flex items-center justify-center w-full">
                    <label for="dropzone-file" class="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600">
                        <div class="flex flex-col items-center justify-center pt-5 pb-6">
                            <p class="mb-2 text-sm text-gray-400"><span class="font-semibold">Dosya seçmek için tıkla</span></p>
                            <p class="text-xs text-gray-500">PDF veya TXT</p>
                        </div>
                        <input id="dropzone-file" name="file" type="file" class="hidden" accept=".pdf, .txt" />
                    </label>
                </div>
                <button type="submit" class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">Yükle ve İşle</button>
            </form>
            {% if message %}
            <div class="mt-4 p-3 bg-blue-900 text-blue-200 rounded text-sm">
                {{ message }}
            </div>
            {% endif %}
        </div>

        <!-- Ayarlar Alanı -->
        <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 class="text-xl font-bold mb-4 text-white border-b border-gray-700 pb-2">⚙️ Ayarlar</h2>
            
            <form action="/admin/settings" method="post" class="space-y-4">
                <div>
                    <label class="block text-sm font-bold mb-1">Model Seçimi</label>
                    <select name="model_name" class="w-full p-2 rounded bg-gray-700 border border-gray-600">
                        <option value="llama3-8b-8192" {% if settings.model == 'llama3-8b-8192' %}selected{% endif %}>Llama 3 8B (Hızlı)</option>
                        <option value="llama3-70b-8192" {% if settings.model == 'llama3-70b-8192' %}selected{% endif %}>Llama 3 70B (Zeki)</option>
                        <option value="mixtral-8x7b-32768" {% if settings.model == 'mixtral-8x7b-32768' %}selected{% endif %}>Mixtral 8x7B</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-bold mb-1">Whitelist (Virgülle Ayırın)</label>
                    <textarea name="whitelist" rows="3" class="w-full p-2 rounded bg-gray-700 border border-gray-600" placeholder="5551234567, 905321112233">{{ settings.whitelist }}</textarea>
                    <p class="text-xs text-gray-500 mt-1">Sadece bu numaralar botu kullanabilir. Boş bırakırsanız herkes kullanır.</p>
                </div>

                <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Ayarları Kaydet</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

# --- Basit Bellek İçi Ayarlar (Persistent olması için DB kullanılabilir) ---
# Gerçek prodüksiyonda bu ayarları Supabase'de bir tabloda tutmalısın.
class Settings:
    model: str = "llama3-70b-8192"
    whitelist: str = "" # Boş ise herkese açık

app_settings = Settings()

# --- Admin Paneli Yardımcıları ---
auth_cookie = APIKeyCookie(name="session_token", auto_error=False)

def verify_admin(cookie: str = Depends(auth_cookie)):
    if cookie != "secret_admin_token":
        return None
    return True

# --- Endpointler: Admin Paneli ---

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, authenticated: bool = Depends(verify_admin)):
    if not authenticated:
        return HTML_LOGIN
    
    # Jinja2 template manuel render (string'den)
    from jinja2 import Template
    template = Template(HTML_DASHBOARD)
    return template.render(message=None, settings=app_settings)

@app.post("/admin/login")
async def login(username: str = Form(...), password: str = Form(...)):
    # BASİT KİMLİK DOĞRULAMA (Değiştirin!)
    if username == "admin" and password == "12345":
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(key="session_token", value="secret_admin_token")
        return response
    return HTMLResponse(content=HTML_LOGIN.replace("Bot Admin Paneli", "Hatalı Giriş!"), status_code=401)

@app.get("/admin/logout")
async def logout():
    response = RedirectResponse(url="/admin", status_code=303)
    response.delete_cookie("session_token")
    return response

@app.post("/admin/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...), authenticated: bool = Depends(verify_admin)):
    if not authenticated:
        return RedirectResponse(url="/admin", status_code=303)
    
    try:
        # Dosyayı geçici olarak kaydet
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # Dosya türüne göre yükle
        docs = []
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_location)
            docs = loader.load()
        else:
            loader = TextLoader(file_location, encoding="utf-8")
            docs = loader.load()

        # Parçalara ayır (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Vektörleri oluştur ve Supabase'e yükle
        vector_store.add_documents(splits)

        # Temizlik
        os.remove(file_location)
        
        msg = f"Başarılı! {len(splits)} parça vektör veritabanına eklendi."
    except Exception as e:
        logger.error(f"Upload error: {e}")
        msg = f"Hata oluştu: {str(e)}"

    from jinja2 import Template
    template = Template(HTML_DASHBOARD)
    return template.render(message=msg, settings=app_settings)

@app.post("/admin/settings", response_class=HTMLResponse)
async def update_settings(model_name: str = Form(...), whitelist: str = Form(""), authenticated: bool = Depends(verify_admin)):
    if not authenticated:
        return RedirectResponse(url="/admin", status_code=303)
    
    app_settings.model = model_name
    app_settings.whitelist = whitelist
    
    from jinja2 import Template
    template = Template(HTML_DASHBOARD)
    return template.render(message="Ayarlar güncellendi.", settings=app_settings)

# --- Endpointler: Webhook & AI Mantığı ---

class WebhookMessage(BaseModel):
    # Evolution API payload yapısı karmaşık olabilir, esnek tutuyoruz
    data: dict = {}
    sender: str = "" # Bu genelde header veya payload içinden gelir

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        body = await request.json()
        data = body.get("data", {})
        
        # Evolution API veri yapısını çözümle
        # Not: Sürüme göre değişebilir. Genelde data.message...
        msg_type = data.get("messageType", "")
        if msg_type != "conversation" and msg_type != "extendedTextMessage":
            return {"status": "ignored_type"}
        
        # Gönderen Numarası (JID)
        remote_jid = data.get("key", {}).get("remoteJid", "")
        sender_phone = remote_jid.split("@")[0]
        
        # Mesaj İçeriği
        message_text = ""
        if "conversation" in data.get("message", {}):
            message_text = data["message"]["conversation"]
        elif "extendedTextMessage" in data.get("message", {}):
            message_text = data["message"]["extendedTextMessage"].get("text", "")
            
        if not message_text:
            return {"status": "no_text"}

        # 1. Whitelist Kontrolü
        if app_settings.whitelist:
            allowed_numbers = [x.strip() for x in app_settings.whitelist.split(",") if x.strip()]
            if sender_phone not in allowed_numbers:
                logger.info(f"Unauthorized access attempt: {sender_phone}")
                return {"status": "unauthorized"}

        # 2. Trigger Kontrolü (@siri ile başlıyorsa veya direkt cevap verilecekse)
        trigger = "@siri"
        if trigger not in message_text.lower():
            # Sadece @siri ile başlayanlara cevap ver
            return {"status": "ignored_no_trigger"}
        
        # Trigger kelimesini temizle
        query = message_text.replace(trigger, "", 1).strip()

        # 3. RAG Pipeline
        logger.info(f"Processing query from {sender_phone}: {query}")
        
        # Supabase Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Groq LLM
        llm = ChatGroq(
            temperature=0, 
            model_name=app_settings.model, 
            groq_api_key=GROQ_API_KEY
        )

        # Prompt
        template = """Aşağıdaki bağlamı kullanarak soruyu cevapla.
        Bilmiyorsan "Bilgim yok" de, uydurma.
        
        Bağlam: {context}
        
        Soru: {question}
        
        Cevap:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response_text = rag_chain.invoke(query)

        # 4. Evolution API'ye Cevap Gönder
        send_url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE}"
        headers = {
            "apikey": EVOLUTION_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "number": remote_jid, # Veya sadece numara
            "text": response_text
        }
        
        # WhatsApp'a gönder
        r = requests.post(send_url, json=payload, headers=headers)
        logger.info(f"Response sent: {r.status_code}")
        
        return {"status": "processed", "reply": response_text}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def health_check():
    return {"status": "running", "service": "RAG Bot"}