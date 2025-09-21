import pandas as pd
from pathlib import Path
import re
import unicodedata
import pandas as pd
import re
import unicodedata
from datetime import datetime
from sentence_transformers import SentenceTransformer
# ==============
# CONFIGURACIÓN
# ==============
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ALQUILER = BASE_DIR / "data" / "raw" / "Alquiler"
RAW_VENTA    = BASE_DIR / "data" / "raw" / "Venta"
PROCESSED    = BASE_DIR / "data" / "processed"
FINAL        = BASE_DIR / "data" / "final"

# Crear carpetas si no existen
for folder in [RAW_ALQUILER, RAW_VENTA, PROCESSED, FINAL]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============
# FUNCIONES
# ==============
def read_csv_safely(p: Path) -> pd.DataFrame:

    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, sep=";")
        except Exception:
            pass
    raise RuntimeError(f"No pude leer {p}")

def parse_distrito_from_filename(p: Path) -> str:
    
    name = p.stem.lower()
    m = re.search(r"distrito_([a-z\-ñ]+)_alquiler", name)
    return m.group(1) if m else None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    cols_map = {
        "precio":"precio",
        "localidad":"localidad",
        "tamaño":"tamanio",
        "tamano":"tamanio",
        "tamanio":"tamanio",
        "habitaciones":"habitaciones",
        "descripcion":"descripcion",
        "link":"link",
        "descripcion_larga":"descripcion_larga",
        "distrito":"distrito",
    }
    expected = ["precio","localidad","tamanio","habitaciones","descripcion","link","descripcion_larga","distrito"]
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: cols_map.get(c, c) for c in df.columns})
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    return df[expected]

def std_text(s: str) -> str:
    
    if pd.isna(s): return s
    s = str(s).lower().strip()
    s = ''.join(
        ch for ch in unicodedata.normalize('NFKD', s)
        if not unicodedata.combining(ch)
    )
    s = s.replace("-", "").replace(" ", "")
    return s
# ==============
# UNIFICAR ALQUILER Y VENTA
# ==============
def unify_files(files, operacion):
    frames = []
    for p in files:
        df = read_csv_safely(p)
        df = normalize_columns(df)
        if df["distrito"].isna().all() or (df["distrito"].astype(str).str.strip() == "").all():
            # Extraer distrito del nombre si no existe
            m = re.search(r"distrito_([a-z\-ñ]+)_" + operacion, p.stem.lower())
            df["distrito"] = m.group(1) if m else None
        else:
            df["distrito"] = df["distrito"].astype(str).str.lower().str.strip()

        df["operacion"] = operacion
        df["origen_archivo"] = p.name
        df["distrito_std"] = df["distrito"].apply(std_text)
        frames.append(df)

    df_final = pd.concat(frames, ignore_index=True)
    if "link" in df_final.columns:
        df_final = df_final.drop_duplicates(subset=["link", "operacion"], keep="first")
    return df_final

files_alq = sorted(RAW_ALQUILER.glob("*.csv"))
files_vta = sorted(RAW_VENTA.glob("*.csv"))

df_alq = unify_files(files_alq, "alquiler")
df_vta = unify_files(files_vta, "venta")

# Guardar intermedios
df_alq.to_csv(PROCESSED / "alquiler_unificado.csv", index=False)
df_vta.to_csv(PROCESSED / "venta_unificado.csv", index=False)
# ==============
# UNIFICARTODO
# ==============
all_cols = sorted(set(df_alq.columns) | set(df_vta.columns))
for df in (df_alq, df_vta):
    for c in all_cols:
        if c not in df.columns:
            df[c] = pd.NA

df_all = pd.concat([df_alq[all_cols], df_vta[all_cols]], ignore_index=True)
df_all = df_all.drop_duplicates(subset=["link", "operacion"], keep="first")
df_all.to_csv(PROCESSED / "inmuebles_unificado_total.csv", index=False)
# ==============
# LIMPIEZA Y VARIABLES EXTRA
# ==============
KEYWORDS = ["antigüedad", "baños", "garaje", "trastero", "piscina", "terraza"]
for kw in KEYWORDS:
    col = f"has_{kw}".replace("ñ", "n")
    df_all[col] = df_all["descripcion_larga"].fillna("").str.lower().apply(lambda x: 1 if kw in x else 0)

def precio_to_float(s):
    if pd.isna(s): return pd.NA
    s = str(s).replace(".", "").replace("€", "").replace(" ", "").replace(",", ".")
    try: return float(s)
    except: return pd.NA

df_all["precio_num"] = df_all["precio"].apply(precio_to_float)
df_all["distrito_std"] = df_all["distrito"].astype(str).map(std_text)

# ==============
# EMBEDDINGS
# ==============
def build_embedding_text(row):
    partes = [
        f"operacion: {row.get('operacion','')}",
        f"distrito: {row.get('distrito','')}",
        f"tamanio_m2: {row.get('tamanio','')}",
        f"habitaciones: {row.get('habitaciones','')}",
        f"banos: {row.get('num_banos','')}",
        f"garaje: {row.get('has_garaje',0)}",
        f"trastero: {row.get('has_trastero',0)}",
        f"piscina: {row.get('has_piscina',0)}",
        f"terraza: {row.get('has_terraza',0)}",
        f"descripcion: {str(row.get('descripcion_larga',''))[:500]}",
    ]
    return " | ".join(map(str, partes))

df_all["texto_embedding"] = df_all.apply(build_embedding_text, axis=1)

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = df_all["texto_embedding"].fillna("").tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
df_all["embedding"] = list(embeddings)

# ==============
# GUARDADO FINAL
# ==============
df_all.to_pickle(FINAL / "inmuebles_unificado_total_final.pkl")
df_all.to_csv(FINAL / "inmuebles_unificado_total_final.csv", index=False)
df_all.to_parquet(FINAL / "inmuebles_unificado_total_final.parquet", index=False)

print("Proceso completado. Archivos en:", FINAL)



