from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import io
import base64
import re
from PIL import Image, ImageDraw
from groq import AsyncGroq

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    groq_client = AsyncGroq()
    app.state.groq_client = groq_client

    mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.state.mongo_client = mongo_client
    app.state.chat_collection = mongo_client["mi_kurzus"]["chat_history"]

    yield
    mongo_client.close()


class ChatInput(BaseModel):
    session_id: str
    prompt: str


app = FastAPI(
    lifespan=lifespan,
    title="RAG, FastAPI és ChromaDB példa (Groq LPU-val)",
    version="1.0.0",
    description="Gyors RAG és Vision megvalósítás Groq LPU és Llama 3 modellek segítségével."
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="kurzus_tudasbazis")


class DocumentInput(BaseModel):
    text: str


class QueryInput(BaseModel):
    prompt: str


@app.post("/add_document")
async def add_document(doc: DocumentInput):
    doc_id = str(uuid.uuid4())
    try:
        collection.add(
            documents=[doc.text],
            ids=[doc_id]
        )
        return {
            "status": "ok",
            "doc_id": doc_id,
            "message": "Dokumentum sikeresen elmentve a tudásbázisba."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_rag")
async def ask_rag(query: QueryInput, request: Request):
    groq_client = request.app.state.groq_client

    try:
        results = collection.query(
            query_texts=[query.prompt],
            n_results=2
        )

        context_list = results.get('documents', [[]])[0]
        context = " ".join(context_list) if context_list else ""

        if not context:
            return {"response": "Nincs elegendő információ az adatbázisban a válaszhoz."}

        augmented_prompt = f"""
        Válaszold meg a felhasználó kérdését a megadott kontextus alapján. 
        Ha a kontextus nem tartalmazza a választ, mondd meg, hogy nem tudod.

        Kontextus: {context}

        Kérdés: {query.prompt}
        """

        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=0.2
        )

        return {
            "retrieved_context": context,
            "llm_response": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_documents")
def list_documents():
    try:
        results = collection.get()
        docs = []
        if results['ids']:
            for i in range(len(results['ids'])):
                docs.append({
                    "id": results['ids'][i],
                    "text": results['documents'][i]
                })

        return {
            "total_documents": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
@app.post("/chat")
async def chat_with_memory(chat_input: ChatInput, request: Request):
    groq_client = request.app.state.groq_client
    chat_coll = request.app.state.chat_collection

    try:
        cursor = chat_coll.find({"session_id": chat_input.session_id}).sort("timestamp", 1)
        history_docs = await cursor.to_list(length=100)

        messages = [{"role": "system", "content": "Segítőkész MI asszisztens vagy."}]
        for doc in history_docs:
            role = "assistant" if doc["role"] in ["model", "assistant"] else "user"
            messages.append({"role": role, "content": doc["content"]})

        messages.append({"role": "user", "content": chat_input.prompt})

        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        reply_text = response.choices[0].message.content
        now = datetime.datetime.utcnow()

        await chat_coll.insert_many([
            {
                "session_id": chat_input.session_id,
                "role": "user",
                "content": chat_input.prompt,
                "timestamp": now
            },
            {
                "session_id": chat_input.session_id,
                "role": "assistant",
                "content": reply_text,
                "timestamp": now + datetime.timedelta(milliseconds=1)
            }
        ])

        return {
            "session_id": chat_input.session_id,
            "response": reply_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat_with_image")
async def chat_with_image(
        request: Request,
        session_id: str = Form(...),
        prompt: str = Form(...),
        file: UploadFile = File(...)
):
    groq_client = request.app.state.groq_client
    chat_coll = request.app.state.chat_collection

    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = file.content_type or "image/jpeg"

        response = await groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )

        reply_text = response.choices[0].message.content
        now = datetime.datetime.utcnow()

        await chat_coll.insert_many([
            {
                "session_id": session_id,
                "role": "user",
                "content": f"[Kép feltöltve: {file.filename}] - {prompt}",
                "timestamp": now
            },
            {
                "session_id": session_id,
                "role": "assistant",
                "content": reply_text,
                "timestamp": now + datetime.timedelta(milliseconds=1)
            }
        ])

        return {
            "session_id": session_id,
            "response": reply_text,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/waldo_search")
async def waldo_search(
        request: Request,
        session_id: str = Form(...),
        prompt: str = Form(...),
        file: UploadFile = File(...)
):
    import base64

    groq_client = request.app.state.groq_client

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = file.content_type or "image/jpeg"

        # Szigorú prompt a koordináták megszerzéséhez
        custom_prompt = (
            f"{prompt}\n\n"
            "FONTOS: A válaszod legvégén feltétlenül add meg a megtalált célpont téglalapját "
            "egyetlen szögletes zárójelben: [ymin, xmin, ymax, xmax].\n"
            "Figyelj a skálára (0 és 1000 között):\n"
            "- ymin=0 a kép TETEJE, ymax=1000 a kép ALJA\n"
            "- xmin=0 a kép BAL széle, xmax=1000 a kép JOBB széle"
        )

        response = await groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }],
            max_tokens=800
        )
        reply_text = response.choices[0].message.content

        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Megkeressük a [ymin, xmin, ymax, xmax] mintát a válaszban
        matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', reply_text)

        if matches:
            ymin, xmin, ymax, xmax = map(int, matches[-1])

            # Biztonsági ellenőrzés és helyes skálázás pixelre
            real_xmin = min(xmin, xmax)
            real_xmax = max(xmin, xmax)
            real_ymin = min(ymin, ymax)
            real_ymax = max(ymin, ymax)

            box = [
                int(real_xmin * width / 1000),
                int(real_ymin * height / 1000),
                int(real_xmax * width / 1000),
                int(real_ymax * height / 1000)
            ]
        else:
            box = [int(width * 0.70), int(height * 0.65), int(width * 0.80), int(height * 0.75)]

        draw.rectangle(box, outline="red", width=5)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "session_id": session_id,
            "response": reply_text,
            "annotated_image": annotated_base64,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
