from contextlib import asynccontextmanager
from typing import TypedDict, cast, Any, List

import asyncio
import base64
import datetime
import io
import re
import uuid

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from groq import AsyncGroq
from motor.core import AgnosticCollection
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image, ImageDraw
from pydantic import BaseModel
from starlette.datastructures import State
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# A Starlette State objektum típusos kiterjesztése.
class AppState(State):
    groq_client: AsyncGroq
    mongo_client: AsyncIOMotorClient
    chat_collection: AgnosticCollection


class TypedFastAPI(FastAPI):
    state: AppState


def get_app_state(request: Request) -> AppState:
    """Típusos hozzáférés a FastAPI alkalmazás állapotához."""
    typed_app = cast(TypedFastAPI, request.app)
    return cast(AppState, typed_app.state)


@asynccontextmanager
async def lifespan(app: TypedFastAPI):
    groq_client = AsyncGroq()
    app.state.groq_client = groq_client

    mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.state.mongo_client = mongo_client
    app.state.chat_collection = mongo_client["mi_kurzus"]["chat_history"]

    try:
        yield
    finally:
        mongo_client.close()
        await groq_client.close()


class ChatInput(BaseModel):
    session_id: str
    prompt: str


app = TypedFastAPI(
    lifespan=lifespan,
    title="RAG, FastAPI és ChromaDB példa (Groq LPU-val)",
    version="1.0.0",
    description=(
        "Gyors RAG és Vision megvalósítás Groq LPU és Quen 3 modellek "
        "segítségével."
    ),
)

waldo_progress: dict[str, dict] = {}

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
        collection.add(documents=[doc.text], ids=[doc_id])
        return {
            "status": "ok",
            "doc_id": doc_id,
            "message": "Dokumentum sikeresen elmentve a tudásbázisba.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask_rag")
async def ask_rag(query: QueryInput, request: Request):
    state = get_app_state(request)
    groq_client = state.groq_client

    try:
        results = collection.query(query_texts=[query.prompt], n_results=2)
        context_list = results.get("documents", [[]])[0]
        context = " ".join(context_list) if context_list else ""

        if not context:
            return {
                "response": (
                    "Nincs elegendő információ az adatbázisban a válaszhoz."
                )
            }

        augmented_prompt = f"""
        Válaszold meg a felhasználó kérdését a megadott kontextus alapján. 
        Ha a kontextus nem tartalmazza a választ, mondd meg, hogy nem tudod.

        Kontextus: {context}

        Kérdés: {query.prompt}
        """

        # ITT VOLT A HIBA - Javítva helyes Groq hívásra:
        response = await cast(Any, groq_client.chat.completions).create(
            model="qwen/qwen3.8-27b",
            messages=cast(Any, [{"role": "user", "content": augmented_prompt}]),
            temperature=0.2,
            max_tokens=800,
        )

        return {
            "retrieved_context": context,
            "llm_response": response.choices[0].message.content,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/list_documents")
def list_documents():
    try:
        results = collection.get()
        docs = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                docs.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                })

        return {"total_documents": len(docs), "documents": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
)
@app.post("/chat")
async def chat_with_memory(chat_input: ChatInput, request: Request):
    state = get_app_state(request)
    groq_client = state.groq_client
    chat_coll = state.chat_collection

    try:
        cursor = chat_coll.find(
            {"session_id": chat_input.session_id}
        ).sort("timestamp", 1)
        history_docs = await cursor.to_list(length=100)

        messages = [
            {"role": "system", "content": "Segítőkész MI asszisztens vagy."}
        ]
        for doc in history_docs:
            role = (
                "assistant"
                if doc["role"] in ["model", "assistant"]
                else "user"
            )
            messages.append({"role": role, "content": doc["content"]})

        messages.append({"role": "user", "content": chat_input.prompt})

        response = await groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=cast(Any, messages)
        )

        reply_text = response.choices[0].message.content
        now = datetime.datetime.utcnow()

        await chat_coll.insert_many([
            {
                "session_id": chat_input.session_id,
                "role": "user",
                "content": chat_input.prompt,
                "timestamp": now,
            },
            {
                "session_id": chat_input.session_id,
                "role": "assistant",
                "content": reply_text,
                "timestamp": now + datetime.timedelta(milliseconds=1),
            },
        ])

        return {"session_id": chat_input.session_id, "response": reply_text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat_with_image")
async def chat_with_image(
    request: Request,
    session_id: str = Form(...),
    prompt: str = Form(...),
    files: List[UploadFile] = File(...),  # Itt változott: fájlok listája
):
    state = get_app_state(request)
    groq_client = state.groq_client
    chat_coll = state.chat_collection

    try:
        # Felépítjük a multimodális üzenet tartalmát (szöveg + összes kép)
        content_items = [{"type": "text", "text": prompt}]
        filenames = []

        for file in files:
            image_bytes = await file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = file.content_type or "image/jpeg"
            filenames.append(file.filename)

            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            })

        response = await cast(Any, groq_client.chat.completions).create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": content_items}],
            max_tokens=800,
        )

        reply_text = response.choices[0].message.content
        now = datetime.datetime.utcnow()
        filenames_str = ", ".join(filenames)

        await chat_coll.insert_many([
            {
                "session_id": session_id,
                "role": "user",
                "content": f"[Képek feltöltve: {filenames_str}] - {prompt}",
                "timestamp": now,
            },
            {
                "session_id": session_id,
                "role": "assistant",
                "content": reply_text,
                "timestamp": now + datetime.timedelta(milliseconds=1),
            },
        ])

        return {
            "session_id": session_id,
            "response": reply_text,
            "filenames": filenames,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/waldo_search")
async def waldo_search(
        request: Request,
        session_id: str = Form(...),
        prompt: str = Form(...),
        file: UploadFile = File(...),
):
    """Waldo keresése közvetlenül a teljes képen."""
    state = get_app_state(request)
    groq_client = state.groq_client

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Üres képfájl érkezett.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="A feltöltött fájl nem olvasható képfájl."
            ) from exc

        width, height = image.size
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"

        waldo_progress[session_id] = {
            "current": 1,
            "total": 1,
            "message": "Kép elemzése a modellel...",
            "done": False,
        }

        # Egységes, teljes képes prompt
        waldo_prompt = (
            f"Feladat: keresd meg Waldo/Wally figurát ezen a teljes képen.\n"
            f"Felhasználói támpont: {prompt}\n\n"
            "Waldo tipikus ismertetőjegye a piros-fehér csíkos sapka, "
            "a szemüveg és a piros-fehér csíkos felső. Csak akkor jelöld "
            "meg, ha tényleg látható. Ne találj ki koordinátákat.\n"
            "A válaszod első sora pontosan az alábbiak egyike legyen:\n"
            "FOUND: NO\n"
            "vagy\n"
            "FOUND: YES [ymin, xmin, ymax, xmax]\n\n"
            "A koordináták a TELJES KÉPRE vonatkozzanak, 0–1000 közötti skálán. "
            "A sorrend: felső, bal, alsó, jobb. A téglalap Waldo teljes alakját fogja körül."
            "Szövegesen is írd le, hogy szerinted hol van Waldo."
            "Ha egy képen piros karika is szerepel, akkor azt külön vizsgáld meg (pl: waldo_solution.jpg)."
        )

        response = await cast(Any, groq_client.chat.completions).create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": waldo_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=200,
            temperature=0,
        )
        reply_text = response.choices[0].message.content or ""

        # Válasz ellenőrzése és kinyerése
        match = re.search(
            r"FOUND\s*:\s*YES\s*\[\s*(\d{1,4})\s*,\s*"
            r"(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]",
            reply_text,
            re.IGNORECASE,
        )

        if not match or re.search(r"FOUND\s*:\s*NO", reply_text, re.IGNORECASE):
            waldo_progress[session_id] = {
                "current": 1,
                "total": 1,
                "message": "A keresés befejeződött, nincs találat.",
                "done": True,
            }
            return {
                "session_id": session_id,
                "response": "A modell nem talált megbízható Waldo-t a képen.",
                "annotated_image": None,
                "found": False,
                "filename": file.filename,
            }

        ymin, xmin, ymax, xmax = map(int, match.groups())

        # Koordináták átváltása pixelbe a teljes kép mérete alapján
        bx1 = round(xmin * width / 1000)
        by1 = round(ymin * height / 1000)
        bx2 = round(xmax * width / 1000)
        by2 = round(ymax * height / 1000)

        box = [bx1, by1, bx2, by2]

        # Kép berajzolása
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(
            box, outline="red", width=max(4, round(min(width, height) / 250))
        )

        buffered = io.BytesIO()
        annotated.save(buffered, format="JPEG", quality=92)
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        waldo_progress[session_id] = {
            "current": 1,
            "total": 1,
            "message": "A keresés befejeződött!",
            "done": True,
        }

        return {
            "session_id": session_id,
            "response": f"A modell megtalálta Waldot!\nModell válasza: {reply_text}",
            "annotated_image": annotated_base64,
            "found": True,
            "box": box,
            "filename": file.filename,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/waldo_progress/{session_id}")
async def get_waldo_progress(session_id: str):
    return waldo_progress.get(
        session_id,
        {
            "current": 0,
            "total": 9,
            "message": "A keresés előkészítése...",
            "done": False,
        },
    )
