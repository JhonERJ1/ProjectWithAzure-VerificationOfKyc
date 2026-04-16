import os
import json
import io
import tempfile
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Azure Clients ──
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

STORAGE_ACCOUNT  = os.getenv("AZURE_STORAGE_ACCOUNT", "kycstoragedev2")
STORAGE_KEY      = os.getenv("AZURE_STORAGE_KEY")
CONN_STR         = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONT_DOCS        = "documentos"
CONT_SELFIES     = "selfies"
CONT_RESULTADOS  = "resultados"

blob_service = BlobServiceClient.from_connection_string(CONN_STR)

doc_cliente = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOC_INTELLIGENCE_KEY"))
)

face_cliente = FaceClient(
    os.getenv("AZURE_FACE_ENDPOINT", "https://centralus.api.cognitive.microsoft.com/"),
    CognitiveServicesCredentials(os.getenv("AZURE_FACE_KEY"))
)

# Ensure result container exists
for cont in [CONT_DOCS, CONT_SELFIES, CONT_RESULTADOS]:
    try:
        blob_service.create_container(cont)
    except Exception:
        pass

EXTENSIONES = {"png", "jpg", "jpeg", "webp"}

# ── Helpers ───

def extension_valida(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in EXTENSIONES

def siguiente_id():
    blobs = list(blob_service.get_container_client(CONT_DOCS).list_blobs())
    docs = [b.name for b in blobs if b.name.endswith("_doc.png")]
    return str(len(docs) + 1).zfill(6)

def generar_url_sas(contenedor, blob_name, hours=1):
    sas = generate_blob_sas(
        account_name=STORAGE_ACCOUNT,
        container_name=contenedor,
        blob_name=blob_name,
        account_key=STORAGE_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=hours)
    )
    return f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{contenedor}/{blob_name}?{sas}"

def subir_imagen(file_obj, contenedor, blob_name):
    file_obj.seek(0)
    blob_service.get_container_client(contenedor).upload_blob(
        blob_name, file_obj, overwrite=True
    )

def guardar_resultado_json(sid, resultado):
    data = json.dumps(resultado, ensure_ascii=False, indent=2).encode("utf-8")
    blob_name = f"resultado_{sid}.json"
    blob_service.get_container_client(CONT_RESULTADOS).upload_blob(
        blob_name, data, overwrite=True
    )
    return blob_name

def extraer_campos_documento(sid):
    url = generar_url_sas(CONT_DOCS, f"onb_sol_{sid}_doc.png")
    poller    = doc_cliente.begin_analyze_document(
        "prebuilt-idDocument",
        AnalyzeDocumentRequest(url_source=url)
    )
    resultado = poller.result()
    campos = {}
    doc_type = None
    for doc in resultado.documents:
        doc_type = doc.doc_type
        for campo, valor in doc.fields.items():
            if valor and valor.content:
                campos[campo] = {
                    "valor":     valor.content,
                    "confianza": round(valor.confidence, 3) if valor.confidence else None
                }
    return {"tipo_documento": doc_type, "campos": campos}

def calcular_confidence_facial(face1, face2):
    def landmarks_a_vector(face):
        lm = face.face_landmarks
        puntos = [
            lm.pupil_left, lm.pupil_right, lm.nose_tip,
            lm.mouth_left, lm.mouth_right,
            lm.eyebrow_left_outer, lm.eyebrow_right_outer,
            lm.under_lip_bottom
        ]
        return [(p.x, p.y) for p in puntos]

    def normalizar(puntos):
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        escala = max(max(xs) - min(xs), max(ys) - min(ys)) or 1
        return [((p[0]-cx)/escala, (p[1]-cy)/escala) for p in puntos]

    v1 = normalizar(landmarks_a_vector(face1))
    v2 = normalizar(landmarks_a_vector(face2))
    distancia = sum(((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5 for a, b in zip(v1, v2)) / len(v1)
    return round(max(0.0, min(1.0, 1 - distancia * 2)), 4)

def verificar_rostros(sid):
    url_doc    = generar_url_sas(CONT_DOCS,    f"onb_sol_{sid}_doc.png")
    url_selfie = generar_url_sas(CONT_SELFIES, f"onb_sol_{sid}_selfie.png")

    rostros_doc = face_cliente.face.detect_with_url(
        url_doc,
        detection_model="detection_01",
        return_face_id=False,
        return_face_landmarks=True,
        return_face_attributes=["headPose", "blur", "exposure"]
    )
    rostros_selfie = face_cliente.face.detect_with_url(
        url_selfie,
        detection_model="detection_01",
        return_face_id=False,
        return_face_landmarks=True,
        return_face_attributes=["headPose", "blur", "exposure"]
    )

    if not rostros_doc:
        raise ValueError("No se detectó rostro en el documento")
    if not rostros_selfie:
        raise ValueError("No se detectó rostro en la selfie")

    confidence = calcular_confidence_facial(rostros_doc[0], rostros_selfie[0])

    blur_selfie = rostros_selfie[0].face_attributes.blur.blur_level if rostros_selfie[0].face_attributes else None

    return {
        "confidence_facial": confidence,
        "rostro_documento":  True,
        "rostro_selfie":     True,
        "blur_selfie":       str(blur_selfie) if blur_selfie else "unknown"
    }

def clasificar(confidence):
    if confidence >= 0.90:
        return "APROBADA_AUTOMATICA"
    elif confidence >= 0.70:
        return "REVISION_MANUAL"
    else:
        return "RECHAZADA"

# ── Routes ──

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    cedula = request.files.get("cedula")
    selfie = request.files.get("selfie")
    nombre = request.form.get("nombre", "").strip()

    if not cedula or not selfie:
        return jsonify({"ok": False, "error": "Debes subir ambos archivos."}), 400
    if not extension_valida(cedula.filename) or not extension_valida(selfie.filename):
        return jsonify({"ok": False, "error": "Formato no soportado. Usa PNG, JPG o WEBP."}), 400

    sid = siguiente_id()
    blob_doc    = f"onb_sol_{sid}_doc.png"
    blob_selfie = f"onb_sol_{sid}_selfie.png"

    # 1. Upload images to Blob Storage
    subir_imagen(cedula, CONT_DOCS,    blob_doc)
    subir_imagen(selfie, CONT_SELFIES, blob_selfie)

    resultado = {
        "solicitud_id":   sid,
        "nombre":         nombre or None,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "blob_documento": blob_doc,
        "blob_selfie":    blob_selfie,
        "documento":      None,
        "facial":         None,
        "clasificacion":  "ERROR",
        "errores":        []
    }

    # 2. Document Intelligence
    try:
        resultado["documento"] = extraer_campos_documento(sid)
    except Exception as e:
        resultado["errores"].append(f"Document Intelligence: {str(e)}")

    # 3. Face verification
    try:
        resultado["facial"] = verificar_rostros(sid)
        resultado["clasificacion"] = clasificar(resultado["facial"]["confidence_facial"])
    except Exception as e:
        resultado["errores"].append(f"Face API: {str(e)}")
        resultado["clasificacion"] = "ERROR_FACIAL"

    # 4. Save JSON to Blob Storage
    json_blob = guardar_resultado_json(sid, resultado)
    resultado["json_blob"] = json_blob

    return jsonify({"ok": True, **resultado})


@app.route("/solicitudes")
def solicitudes():
    try:
        blobs = list(blob_service.get_container_client(CONT_RESULTADOS).list_blobs())
        ids = sorted([
            b.name.replace("resultado_", "").replace(".json", "")
            for b in blobs if b.name.startswith("resultado_")
        ])
        return jsonify({"total": len(ids), "ids": ids})
    except Exception as e:
        return jsonify({"total": 0, "ids": [], "error": str(e)})


@app.route("/solicitud/<sid>")
def detalle_solicitud(sid):
    try:
        blob_name = f"resultado_{sid}.json"
        data = blob_service.get_container_client(CONT_RESULTADOS).download_blob(blob_name).readall()
        return jsonify(json.loads(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)