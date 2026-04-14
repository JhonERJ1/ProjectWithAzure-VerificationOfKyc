import os
import shutil
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

CARPETA_DOCS    = "datos_crudos"
EXTENSIONES     = {"png", "jpg", "jpeg", "webp"}

os.makedirs(CARPETA_DOCS, exist_ok=True)

def siguiente_id():
    existentes = [
        f for f in os.listdir(CARPETA_DOCS)
        if f.startswith("onb_sol_") and f.endswith("_doc.png")
    ]
    return str(len(existentes) + 1).zfill(6)

def extension_valida(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in EXTENSIONES

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    cedula = request.files.get("cedula")
    selfie = request.files.get("selfie")

    if not cedula or not selfie:
        return jsonify({"ok": False, "error": "Debes subir ambos archivos."}), 400

    if not extension_valida(cedula.filename) or not extension_valida(selfie.filename):
        return jsonify({"ok": False, "error": "Formato no soportado. Usa PNG, JPG o WEBP."}), 400

    sid = siguiente_id()

    cedula.save(os.path.join(CARPETA_DOCS, f"onb_sol_{sid}_doc.png"))
    selfie.save(os.path.join(CARPETA_DOCS, f"onb_sol_{sid}_selfie.png"))

    return jsonify({
        "ok": True,
        "solicitud_id": sid,
        "doc":    f"onb_sol_{sid}_doc.png",
        "selfie": f"onb_sol_{sid}_selfie.png",
    })

@app.route("/solicitudes")
def solicitudes():
    docs = sorted([
        f.replace("_doc.png", "").replace("onb_sol_", "")
        for f in os.listdir(CARPETA_DOCS)
        if f.endswith("_doc.png")
    ])
    return jsonify({"total": len(docs), "ids": docs})

if __name__ == "__main__":
    app.run(debug=True, port=5000)