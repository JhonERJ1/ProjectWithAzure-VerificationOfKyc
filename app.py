import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, send_from_directory, make_response
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

# ── Cookie de sesión anónima ──
COOKIE_USER_ID   = "kyc_user_id"
COOKIE_MAX_AGE   = 60 * 60 * 24 * 365  # 1 año

blob_service = BlobServiceClient.from_connection_string(CONN_STR)

doc_cliente = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOC_INTELLIGENCE_KEY"))
)

face_cliente = FaceClient(
    os.getenv("AZURE_FACE_ENDPOINT", "https://centralus.api.cognitive.microsoft.com/"),
    CognitiveServicesCredentials(os.getenv("AZURE_FACE_KEY"))
)

# Se asegura de que existan los contenedores
for cont in [CONT_DOCS, CONT_SELFIES, CONT_RESULTADOS]:
    try:
        blob_service.create_container(cont)
    except Exception:
        pass

EXTENSIONES = {"png", "jpg", "jpeg", "webp"}
TAMANO_MAXIMO_MB = 10
TAMANO_MAXIMO_BYTES = TAMANO_MAXIMO_MB * 1024 * 1024

# Campos típicos de una cédula / documento de identidad que debería detectar
# Document Intelligence con el modelo prebuilt-idDocument
CAMPOS_DOCUMENTO_VALIDO = {
    "FirstName", "LastName", "DocumentNumber", "DateOfBirth",
    "DateOfExpiration", "Sex", "Nationality", "CountryRegion",
    "Address", "PersonalNumber", "MachineReadableZone"
}

# ── Helpers ──

def obtener_user_id():
    """Obtiene el user_id de la cookie. Si no existe, retorna None (lo crea /upload)."""
    return request.cookies.get(COOKIE_USER_ID)

def extension_valida(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in EXTENSIONES

def tamano_archivo(file_obj):
    """Obtiene el tamaño del archivo sin consumirlo."""
    file_obj.seek(0, 2)  # al final
    size = file_obj.tell()
    file_obj.seek(0)     # regresar al inicio
    return size

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

def eliminar_blob(contenedor, blob_name):
    """Elimina un blob. Usado para limpieza cuando la validación falla."""
    try:
        blob_service.get_container_client(contenedor).delete_blob(blob_name)
    except Exception:
        pass

def guardar_resultado_json(sid, resultado):
    data = json.dumps(resultado, ensure_ascii=False, indent=2).encode("utf-8")
    blob_name = f"resultado_{sid}.json"
    blob_service.get_container_client(CONT_RESULTADOS).upload_blob(
        blob_name, data, overwrite=True
    )
    return blob_name

# ── VALIDACIÓN: detectar si los archivos van en el campo correcto ──

def detectar_rostros_simple(url_imagen):
    """Detecta rostros en una imagen y retorna la lista."""
    try:
        return face_cliente.face.detect_with_url(
            url_imagen,
            detection_model="detection_03",
            return_face_id=False,
            return_face_landmarks=False,
            return_face_attributes=[]
        )
    except Exception:
        return []

def analizar_documento_identidad(url_imagen):
    """
    Analiza una imagen con el modelo de cédula.
    Retorna (campos_detectados, es_documento_valido)
    """
    try:
        poller = doc_cliente.begin_analyze_document(
            "prebuilt-idDocument",
            AnalyzeDocumentRequest(url_source=url_imagen)
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
        # Se considera documento válido si detecta al menos 2 campos
        # reconocidos de un documento de identidad
        campos_reconocidos = set(campos.keys()) & CAMPOS_DOCUMENTO_VALIDO
        es_valido = len(campos_reconocidos) >= 2
        return {"tipo_documento": doc_type, "campos": campos, "es_valido": es_valido}
    except Exception as e:
        return {"tipo_documento": None, "campos": {}, "es_valido": False, "error": str(e)}

def validar_campos_antes_de_procesar(url_doc, url_selfie):
    """
    Valida que:
    - La imagen del 'documento' sea efectivamente un documento de identidad
      (Document Intelligence extrae campos reconocibles)
    - La imagen de la 'selfie' contenga un rostro y NO sea un documento
      (no debería extraerse como cédula)

    Retorna (ok: bool, errores: list, doc_analizado: dict)
    """
    errores = []

    # Análisis del supuesto documento
    doc_analizado = analizar_documento_identidad(url_doc)
    rostros_doc   = detectar_rostros_simple(url_doc)

    # Análisis de la supuesta selfie
    rostros_selfie = detectar_rostros_simple(url_selfie)
    # Verificamos si la selfie es en realidad un documento
    selfie_como_doc = analizar_documento_identidad(url_selfie)

    # Regla 1: el documento debe parecer una cédula
    if not doc_analizado["es_valido"]:
        errores.append(
            "La imagen del documento no parece una cédula o documento de identidad válido. "
            "Asegúrate de subir una foto clara del frente de tu cédula."
        )

    # Regla 2: el documento debería tener un rostro
    if not rostros_doc:
        errores.append(
            "No se detectó un rostro en la imagen del documento. "
            "Verifica que la cédula esté bien enfocada y legible."
        )

    # Regla 3: la selfie debe tener exactamente un rostro
    if not rostros_selfie:
        errores.append(
            "No se detectó un rostro en la selfie. "
            "Asegúrate de que tu cara esté visible y bien iluminada."
        )
    elif len(rostros_selfie) > 1:
        errores.append(
            f"Se detectaron {len(rostros_selfie)} rostros en la selfie. "
            "La foto debe ser solo tuya, sin otras personas."
        )

    # Regla 4: la selfie NO debería ser un documento
    if selfie_como_doc["es_valido"]:
        errores.append(
            "La imagen de la selfie parece ser un documento de identidad. "
            "¿Invertiste los archivos? La selfie debe ser una foto de tu rostro, no una cédula."
        )

    return (len(errores) == 0, errores, doc_analizado)

# ── Extracción y verificación (ya no analizamos dos veces) ──

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
    response = make_response(send_from_directory("templates", "index.html"))
    # Si el usuario no tiene cookie, le asignamos una nueva
    if not request.cookies.get(COOKIE_USER_ID):
        nuevo_id = uuid.uuid4().hex
        response.set_cookie(
            COOKIE_USER_ID,
            nuevo_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax"
        )
    return response

@app.route("/upload", methods=["POST"])
def upload():
    # ── Asegurar user_id ──
    user_id = obtener_user_id()
    nuevo_user_id = None
    if not user_id:
        user_id = uuid.uuid4().hex
        nuevo_user_id = user_id

    cedula = request.files.get("cedula")
    selfie = request.files.get("selfie")
    nombre = request.form.get("nombre", "").strip()

    # ── Validaciones rápidas ──
    if not cedula or not selfie:
        return jsonify({"ok": False, "error": "Debes subir ambos archivos."}), 400
    if not extension_valida(cedula.filename) or not extension_valida(selfie.filename):
        return jsonify({"ok": False, "error": "Formato no soportado. Usa PNG, JPG o WEBP."}), 400

    # Validación de tamaño
    if tamano_archivo(cedula) > TAMANO_MAXIMO_BYTES:
        return jsonify({"ok": False, "error": f"La imagen del documento excede {TAMANO_MAXIMO_MB} MB."}), 400
    if tamano_archivo(selfie) > TAMANO_MAXIMO_BYTES:
        return jsonify({"ok": False, "error": f"La selfie excede {TAMANO_MAXIMO_MB} MB."}), 400

    sid = siguiente_id()
    blob_doc    = f"onb_sol_{sid}_doc.png"
    blob_selfie = f"onb_sol_{sid}_selfie.png"

    # 1. Sube imágenes al Blob storage
    subir_imagen(cedula, CONT_DOCS,    blob_doc)
    subir_imagen(selfie, CONT_SELFIES, blob_selfie)

    # 2. VALIDACIÓN CRUZADA: ¿están los archivos en el campo correcto?
    url_doc    = generar_url_sas(CONT_DOCS,    blob_doc)
    url_selfie = generar_url_sas(CONT_SELFIES, blob_selfie)

    ok_validacion, errores_validacion, doc_analizado = validar_campos_antes_de_procesar(
        url_doc, url_selfie
    )

    if not ok_validacion:
        # Si la validación falla eliminamos las imágenes subidas
        # para no dejar basura en el storage
        eliminar_blob(CONT_DOCS,    blob_doc)
        eliminar_blob(CONT_SELFIES, blob_selfie)
        resp = jsonify({
            "ok": False,
            "error": "Validación de archivos falló.",
            "errores": errores_validacion,
            "validacion_fallida": True
        })
        if nuevo_user_id:
            resp.set_cookie(COOKIE_USER_ID, nuevo_user_id,
                            max_age=COOKIE_MAX_AGE, httponly=True, samesite="Lax")
        return resp, 400

    # 3. Construir resultado (ya tenemos el análisis del documento, lo reutilizamos)
    resultado = {
        "solicitud_id":   sid,
        "user_id":        user_id,
        "nombre":         nombre or None,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "blob_documento": blob_doc,
        "blob_selfie":    blob_selfie,
        "documento":      {
            "tipo_documento": doc_analizado["tipo_documento"],
            "campos":         doc_analizado["campos"]
        },
        "facial":         None,
        "clasificacion":  "ERROR",
        "errores":        []
    }

    # 4. Verificación facial (landmarks y comparación)
    try:
        resultado["facial"] = verificar_rostros(sid)
        resultado["clasificacion"] = clasificar(resultado["facial"]["confidence_facial"])
    except Exception as e:
        resultado["errores"].append(f"Face API: {str(e)}")
        resultado["clasificacion"] = "ERROR_FACIAL"

    # 5. Guarda el JSON en Blob storage
    json_blob = guardar_resultado_json(sid, resultado)
    resultado["json_blob"] = json_blob

    resp = jsonify({"ok": True, **resultado})
    if nuevo_user_id:
        resp.set_cookie(COOKIE_USER_ID, nuevo_user_id,
                        max_age=COOKIE_MAX_AGE, httponly=True, samesite="Lax")
    return resp


@app.route("/solicitudes")
def solicitudes():
    """Lista solo las solicitudes del usuario actual (filtradas por user_id)."""
    user_id = obtener_user_id()
    if not user_id:
        return jsonify({"total": 0, "ids": []})

    try:
        blobs = list(blob_service.get_container_client(CONT_RESULTADOS).list_blobs())
        ids_usuario = []
        for b in blobs:
            if not b.name.startswith("resultado_") or not b.name.endswith(".json"):
                continue
            try:
                data = blob_service.get_container_client(CONT_RESULTADOS).download_blob(b.name).readall()
                resultado = json.loads(data)
                # Solo incluir si pertenece al usuario actual
                if resultado.get("user_id") == user_id:
                    sid = b.name.replace("resultado_", "").replace(".json", "")
                    ids_usuario.append(sid)
            except Exception:
                continue
        ids_usuario.sort()
        return jsonify({"total": len(ids_usuario), "ids": ids_usuario})
    except Exception as e:
        return jsonify({"total": 0, "ids": [], "error": str(e)})


@app.route("/solicitud/<sid>")
def detalle_solicitud(sid):
    """Solo muestra el detalle si pertenece al usuario actual."""
    user_id = obtener_user_id()
    if not user_id:
        return jsonify({"error": "Sesión no válida"}), 403

    try:
        blob_name = f"resultado_{sid}.json"
        data = blob_service.get_container_client(CONT_RESULTADOS).download_blob(blob_name).readall()
        resultado = json.loads(data)

        # Verificar que el resultado pertenece al usuario
        if resultado.get("user_id") != user_id:
            return jsonify({"error": "No tienes permiso para ver esta solicitud"}), 403

        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    app.run()
