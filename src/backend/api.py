from contextlib import asynccontextmanager
from typing import TypedDict, cast, Any, List

import asyncio
import base64
import datetime
import io
import os
from pathlib import Path
import re
import uuid

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, FileResponse
from groq import AsyncGroq
from motor.core import AgnosticCollection
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image, ImageDraw
from pydantic import BaseModel
from starlette.datastructures import State
from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import AsyncGraphDatabase
import json

load_dotenv()

# Feltöltött képek mappája a szerveren
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AppState(State):
    groq_client: AsyncGroq
    mongo_client: AsyncIOMotorClient
    chat_collection: AgnosticCollection


class TypedFastAPI(FastAPI):
    state: AppState

class GraphDocumentInput(BaseModel):
    text: str

def get_app_state(request: Request) -> AppState:
    typed_app = cast(TypedFastAPI, request.app)
    return cast(AppState, typed_app.state)

class DeleteRelationRequest(BaseModel):
    source: str
    target: str
    relationship: str

@asynccontextmanager
async def lifespan(app: TypedFastAPI):
    groq_client = AsyncGroq()
    app.state.groq_client = groq_client

    mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.state.mongo_client = mongo_client
    app.state.chat_collection = mongo_client["mi_kurzus"]["chat_history"]

    # Neo4j csatlakozás hozzáadása
    neo4j_driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password123")
    )
    app.state.neo4j_driver = neo4j_driver

    try:
        yield
    finally:
        mongo_client.close()
        await groq_client.close()
        await neo4j_driver.close()


class ChatInput(BaseModel):
    session_id: str
    prompt: str


app = TypedFastAPI(
    lifespan=lifespan,
    title="RAG, FastAPI és ChromaDB példa (Groq LPU-val)",
    version="1.1.0",
    description=(
        "Gyors RAG és Multimodális megvalósítás Groq LPU és Qwen modellek "
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
        collection.add(
            documents=[doc.text],
            ids=[doc_id],
            metadatas=[{"type": "text"}]
        )
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
        results = collection.query(
            query_texts=[query.prompt],
            n_results=2,
            include=["documents", "metadatas"]
        )
        context_list = results.get("documents", [[]])[0]
        metadata_list = results.get("metadatas", [[]])[0]

        context_text = " ".join(context_list) if context_list else ""

        if not context_text:
            return {
                "response": "Nincs elegendő információ az adatbázisban a válaszhoz.",
                "matched_images": []
            }

        content_items = []
        augmented_prompt = f"""
        Válaszold meg a felhasználó kérdését a megadott kontextus alapján (szövegek és mellékelt képek). 
        Ha a kontextus nem tartalmazza a választ, mondd meg, hogy nem tudod.

        Kontextus szöveg: {context_text}

        Kérdés: {query.prompt}
        """
        content_items.append({"type": "text", "text": augmented_prompt})

        matched_images = []

        for meta in metadata_list:
            if meta and meta.get("type") == "image" and "filepath" in meta:
                file_path = meta["filepath"]
                if os.path.exists(file_path):
                    matched_images.append(os.path.basename(file_path))
                    with open(file_path, "rb") as img_file:
                        img_bytes = img_file.read()
                        b64_img = base64.b64encode(img_bytes).decode("utf-8")
                        content_items.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })

        response = await cast(Any, groq_client.chat.completions).create(
            model="qwen/qwen3.8-27b",
            messages=cast(Any, [{"role": "user", "content": content_items}]),
            temperature=0.2,
            max_tokens=800,
        )

        return {
            "retrieved_context": context_text,
            "llm_response": response.choices[0].message.content,
            "matched_images": matched_images,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/list_documents")
def list_documents():
    try:
        results = collection.get(include=["documents", "metadatas"])
        docs = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i] if results["metadatas"] and i < len(results["metadatas"]) else {}
                docs.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": metadata
                })

        return {"total_documents": len(docs), "documents": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/get_image/{filename}", include_in_schema=False)
def get_uploaded_image(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="A kép nem található.")


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
        files: List[UploadFile] = File(...),
):
    state = get_app_state(request)
    groq_client = state.groq_client
    chat_coll = state.chat_collection

    try:
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
            max_tokens=900,
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

        waldo_prompt = (
            f"Feladat: keresd meg Waldo/Wally figurát ezen a teljes képen.\n"
            f"Felhasználói támpont: {prompt}\n\n"
            "Waldo tipikus ismertetőjegye a piros-fehér csíkos sapka, a barna színű haj, "
            "a fekete szemüveg és a piros-fehér csíkos felső. Olykor csak a feje látható.\n"
            "A válaszod első sora pontosan az alábbiak egyike legyen:\n"
            "FOUND: NO\n"
            "vagy\n"
            "FOUND: YES [ymin, xmin, ymax, xmax]\n\n"
            "A koordináták a TELJES KÉPRE vonatkozzanak, 0–1000 közötti skálán. "
            "A sorrend: felső, bal, alsó, jobb."
            "Ha olyan képet kapsz, amire ráhhelyeztek Paint programmal egy piros kört, akkor a benne lévő részt külön vizsgáld meg."
            "Jellemezd szerinted mit ábrázol a kép"
            "Ha megtaláltad Waldot akkor szövegesen is írd le hol találtad meg"
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

        bx1 = round(xmin * width / 1000)
        by1 = round(ymin * height / 1000)
        bx2 = round(xmax * width / 1000)
        by2 = round(ymax * height / 1000)

        box = [bx1, by1, bx2, by2]

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


@app.post("/add_image_rag")
async def add_image_rag(
        request: Request,
        description: str = Form(""),
        files: List[UploadFile] = File(...)
):
    allowed_types = ["image/jpeg", "image/jpg"]
    state = get_app_state(request)
    groq_client = state.groq_client

    added_documents = []

    try:
        for file in files:
            filename_lower = file.filename.lower() if file.filename else ""
            if file.content_type not in allowed_types and not filename_lower.endswith((".jpg", ".jpeg")):
                raise HTTPException(
                    status_code=400,
                    detail=f"Csak JPG vagy JPEG formátumú képek tölthetők fel! Helytelen fájl: {file.filename}"
                )

            image_bytes = await file.read()
            if not image_bytes:
                continue

            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / unique_filename
            with open(file_path, "wb") as buffer:
                buffer.write(image_bytes)

            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = file.content_type or "image/jpeg"

            final_description = description.strip()
            if not final_description:
                vision_prompt = (
                    "Jellemezd részletesen és pontosan ezt a képet magyarul, "
                    "hogy egy tudásbázisban tárolva hatékonyan kereshető legyen."
                )
                vision_response = await cast(Any, groq_client.chat.completions).create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=2000,
                )
                final_description = vision_response.choices[0].message.content or "Kép leírás nem elérhető."

            doc_id = str(uuid.uuid4())

            collection.add(
                documents=[f"[Kép - {file.filename}]: {final_description}"],
                ids=[doc_id],
                metadatas=[{
                    "filename": file.filename,
                    "type": "image",
                    "filepath": str(file_path)
                }]
            )

            added_documents.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "generated_description": final_description
            })

        return {
            "status": "ok",
            "added_count": len(added_documents),
            "documents": added_documents,
            "message": "A képek sikeresen elemezve és rögzítve lettek a tudásbázisban!"
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detect_and_crop")
async def detect_and_crop(
        request: Request,
        session_id: str = Form(...),
        prompt: str = Form(...),
        file: UploadFile = File(...),
):
    allowed_types = ["image/jpeg", "image/jpg"]
    filename_lower = file.filename.lower() if file.filename else ""
    if file.content_type not in allowed_types and not filename_lower.endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Csak JPG vagy JPEG formátumú képek tölthetők fel!")

    state = get_app_state(request)
    groq_client = state.groq_client

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Üres képfájl érkezett.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="A feltöltött fájl nem olvasható képfájl.") from exc

        width, height = image.size
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"

        detection_prompt = (
            f"Feladat: keresd meg a következő objektumot/elemet a képen: '{prompt}'.\n"
            "A válaszod első sora pontosan az alábbiak egyike legyen:\n"
            "FOUND: NO\n"
            "vagy\n"
            "FOUND: YES [ymin, xmin, ymax, xmax]\n\n"
            "A koordináták a TELJES KÉPRE vonatkozzanak, 0–1000 közötti skálán. "
            "A sorrend: felső, bal, alsó, jobb."
        )

        response = await cast(Any, groq_client.chat.completions).create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300,
            temperature=0,
        )
        reply_text = response.choices[0].message.content or ""

        match = re.search(
            r"FOUND\s*:\s*YES\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]",
            reply_text,
            re.IGNORECASE,
        )

        if not match or re.search(r"FOUND\s*:\s*NO", reply_text, re.IGNORECASE):
            return {
                "session_id": session_id,
                "response": "A modell nem találta meg a keresett objektumot a képen.",
                "found": False,
                "annotated_image": None,
                "cropped_image": None,
            }

        ymin, xmin, ymax, xmax = map(int, match.groups())

        bx1 = round(xmin * width / 1000)
        by1 = round(ymin * height / 1000)
        bx2 = round(xmax * width / 1000)
        by2 = round(ymax * height / 1000)
        box = [bx1, by1, bx2, by2]

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(box, outline="red", width=max(4, round(min(width, height) / 250)))

        buffered_annotated = io.BytesIO()
        annotated.save(buffered_annotated, format="JPEG", quality=92)
        annotated_base64 = base64.b64encode(buffered_annotated.getvalue()).decode("utf-8")

        cropped = image.crop((bx1, by1, bx2, by2))
        buffered_cropped = io.BytesIO()
        cropped.save(buffered_cropped, format="JPEG", quality=92)
        cropped_base64 = base64.b64encode(buffered_cropped.getvalue()).decode("utf-8")

        return {
            "session_id": session_id,
            "response": f"Sikeres detektálás!\n{reply_text}",
            "found": True,
            "box": box,
            "annotated_image": annotated_base64,
            "cropped_image": cropped_base64,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/get_chats/{session_id}")
async def get_chats(session_id: str, request: Request):
    state = get_app_state(request)
    chat_coll = state.chat_collection
    try:
        # Lekérdjük az adott session üzeneteit időrendben
        cursor = chat_coll.find({"session_id": session_id}).sort("timestamp", 1)
        docs = await cursor.to_list(length=100)

        # Az ObjectId-t át kell alakítani string-gé a JSON válasz miatt
        for doc in docs:
            doc["_id"] = str(doc["_id"])

        return {"session_id": session_id, "messages": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/get_all_chats")
async def get_all_chats(request: Request):
    state = get_app_state(request)
    chat_coll = state.chat_collection
    try:
        cursor = chat_coll.find({}).sort("timestamp", -1).limit(100)
        docs = await cursor.to_list(length=100)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return {"total": len(docs), "messages": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/extract_graph")
async def extract_graph(doc: GraphDocumentInput, request: Request):
    """Gráf alapú Tudáskinyerés (GraphRAG) Groq / Qwen segítségével"""
    state = get_app_state(request)
    groq_client = state.groq_client
    neo4j_driver = state.neo4j_driver

    extraction_prompt = f"""
    Az alábbi szövegből nyerj ki entitásokat (személyek, helyszínek, technológiák, fogalmak) és a köztük lévő logikai vagy cselekvési kapcsolatokat. 
    A kimeneted KIZÁRÓLAG egy érvényes JSON tömb (array) legyen, semmi más szöveg, markdown kódblokk se (ne használj ```json ... ``` taget, csak tiszta JSON szöveget!).
    A JSON tömb minden eleme egy objektum legyen ezzel a struktúrával:
    {{
        "source": "Entitás 1 neve",
        "source_label": "Entitás 1 típusa (pl. Person, Technology, Concept)",
        "target": "Entitás 2 neve",
        "target_label": "Entitás 2 típusa",
        "relationship": "KAPCSOLAT_TÍPUSA (Csupa nagybetűvel, pl. HASZNÁLJA, FELESÉGE, RÉSZE)"
    }}

    Szöveg:
    {doc.text}
    """

    try:
        response = await cast(Any, groq_client.chat.completions).create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        raw_text = response.choices[0].message.content.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3]

        relations = json.loads(raw_text.strip())

        async with neo4j_driver.session() as session:
            for rel in relations:
                query = f"""
                MERGE (a:`{rel['source_label']}` {{name: $source_name}})
                MERGE (b:`{rel['target_label']}` {{name: $target_name}})
                MERGE (a)-[r:`{rel['relationship']}`]->(b)
                """
                await session.run(
                    query,
                    source_name=rel["source"],
                    target_name=rel["target"]
                )

        return {
            "status": "ok",
            "message": f"{len(relations)} reláció sikeresen kinyerve és elmentve a gráfba.",
            "extracted_data": relations
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/get_graph")
async def get_graph(request: Request):
    """Minden elmentett reláció lekérdezése a Neo4j gráfadatbázisból"""
    state = get_app_state(request)
    neo4j_driver = state.neo4j_driver

    try:
        async with neo4j_driver.session() as session:
            # Cypher lekérdezés az összes kapcsolat és csomópont lekérésére
            query = """
            MATCH (a)-[r]->(b)
            RETURN a.name AS source, labels(a)[0] AS source_label, 
                   type(r) AS relationship, 
                   b.name AS target, labels(b)[0] AS target_label
            """
            result = await session.run(query)
            records = await result.data()

            relations = []
            for record in records:
                relations.append({
                    "source": record["source"],
                    "source_label": record["source_label"],
                    "target": record["target"],
                    "target_label": record["target_label"],
                    "relationship": record["relationship"]
                })

        return {
            "status": "ok",
            "total_relations": len(relations),
            "extracted_data": relations
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/delete_graph_relation")
async def delete_graph_relation(request: Request, data: DeleteRelationRequest):
    state = get_app_state(request)
    neo4j_driver = state.neo4j_driver

    async with neo4j_driver.session() as session:
        query = """
        MATCH (a {name: $source})-[r]->(b {name: $target})
        WHERE type(r) = $relationship
        DELETE r
        """
        await session.run(query, source=data.source, target=data.target, relationship=data.relationship)

    return {"status": "ok", "message": "Kapcsolat sikeresen törölve a gráfból."}

@app.get("/get_sessions")
async def get_sessions(request: Request):
    """Minden egyedi session_id lekérdezése a MongoDB-ből"""
    state = get_app_state(request)
    chat_coll = state.chat_collection
    try:
        sessions = await chat_coll.distinct("session_id")
        return {"status": "ok", "sessions": sessions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
