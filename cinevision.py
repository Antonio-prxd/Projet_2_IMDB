# ==========================================================
# APP STREAMLIT
# ==========================================================

# --------------------
# IMPORTS
# --------------------
import os
import io
import zipfile
import base64
from pathlib import Path
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import math

from streamlit_option_menu import option_menu

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Cinévision | Films & Recommandations", page_icon="🎬", layout="wide")

DATA_PATH = "https://raw.githubusercontent.com/Antonio-prxd/Projet_2_IMDB/master/FinalMerge.csv"


# Dossier des KPI (captures)
KPI_DIR = os.path.join("assets", "kpis")
KPI_EXT = (".png", ".jpg", ".jpeg", ".webp")

# Poids ML
POIDS_CHOIX = ["très faible", "faible", "moyen", "fort", "très fort"]
POIDS = {
    "très faible": 0.2,
    "faible": 0.5,
    "moyen": 1.0,
    "fort": 2.0,
    "très fort": 5.0
}

# --------------------
# DATA LOADER
# --------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

# --------------------
# IMAGE LOADER
# --------------------
BASE_DIR = Path(__file__).resolve().parent
IMG_HOME = "https://raw.githubusercontent.com/Antonio-prxd/Projet_2_IMDB/refs/heads/main/assets/images/Image_page_acceuil.png.png"

def _img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# UI UTILITIES — CSS GLOBAL + COMPOSANTS
# ============================================================

def inject_global_css():
    """Injecte le CSS global CineData-style (1 seule injection, en tête de main())."""
    st.markdown(
        """
        <style>
        /* ─── GOOGLE FONTS ─── */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Inter:wght@400;500;600;700&display=swap');

        /* ─── CSS VARIABLES ─── */
        :root {
            --gold:        #FFD700;
            --violet:      #9B59F5;
            --violet-glow: rgba(155,89,245,0.40);
            --accent:      #7B2FBE;
            --accent2:     #3D5AFE;
            --bg-dark:     #08081e;
            --surface:     rgba(255,255,255,0.05);
            --border:      rgba(155,89,245,0.30);
            --border-dim:  rgba(255,255,255,0.10);
            --radius:      12px;
            --radius-lg:   18px;
            --shadow:      0 8px 32px rgba(0,0,0,0.55);
            --font-title:  'Cinzel', serif;
            --font-body:   'Inter', sans-serif;
        }

        /* ─── APP BASE ─── */
        html, body, .stApp {
            font-family: var(--font-body);
            color: var(--text-color, #e8e8f0);
        }

        /* ─── HIDE DEFAULT STREAMLIT CHROME ─── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent !important;}

        /* ─── SIDEBAR : verrouillage + fond transparent ─── */
        section[data-testid="stSidebar"] {
            background: rgba(10, 10, 30, 0.65) !important; /* Transparence pour voir l'image de fond */
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-right: 1px solid rgba(155,89,245,0.22) !important;
            min-width: 250px;
        }
        /* Force text to stay white in the dark sidebar regardless of theme */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] > div:first-child {
            background: transparent !important;
        }
        /* FIX: padding du contenu de sidebar (sinon tout reste “collé à gauche”) */
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
            padding: 14px 14px 18px !important;
        }
        /* Cacher le bouton collapse de la sidebar */
        button[data-testid="baseButton-headerNoPadding"],
        [data-testid="stSidebarCollapseButton"],
        button[kind="header"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] * {
            font-family: var(--font-body);
        }

        /* ─────────────────────────────────────────────
           CARDS & MODALS OVERLAYS — CINEDATA PREMIUM
           ───────────────────────────────────────────── */
        .movie-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 24px;
            position: relative;
        }
        .movie-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(155, 89, 245, 0.4);
            border-color: rgba(155, 89, 245, 0.5);
        }

        /* ─── BOUTON RECO PREMIUM ─── */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #9B59F5 0%, #6d28d9 100%) !important;
            border: none !important;
            color: white !important;
            padding: 24px 0 !important;
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(109, 40, 217, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(109, 40, 217, 0.6) !important;
            background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%) !important;
        }

        /* Cache les boutons Streamlit qui servent de trigger sans casser le clic */
        .movie-card + div button {
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            opacity: 0 !important;
            z-index: 20 !important;
            padding: 0 !important;
            border: none !important;
        }
        .movie-poster-container {
            width: 100%;
            aspect-ratio: 2/3;
            overflow: hidden;
            position: relative;
        }
        .movie-poster-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .movie-gradient {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 80%;
            background: linear-gradient(to top, rgba(10,10,25,0.95) 0%, rgba(10,10,25,0.6) 40%, transparent 100%);
        }
        .movie-info {
            position: absolute;
            bottom: 0;
            left: 0; right: 0;
            padding: 16px;
        }
        .movie-title {
            font-family: var(--font-body);
            font-weight: 800; font-size: 0.95rem; color: white;
            margin-bottom: 6px; line-height: 1.2;
            text-shadow: 0 2px 4px rgba(0,0,0,0.8);
            display: -webkit-box; -webkit-line-clamp: 2;
            -webkit-box-orient: vertical; overflow: hidden;
        }
        .movie-meta {
            font-size: 0.8rem; color: #ccd0e0; display: flex; align-items: center; gap: 8px; font-weight: 600;
        }
        .movie-votes {
            font-size: 0.75rem; color: #8890cc; margin-top: 4px; display: flex; align-items: center; gap: 4px;
        }

        /* ──────────────────────────────────────────────
           SIDEBAR NAV — boutons personnalisés
           ────────────────────────────────────────────── */

        /* Style de base pour les boutons nav sidebar */
        section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(155,89,245,0.25) !important;
            border-radius: 12px !important;
            color: rgba(220,224,245,0.92) !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            width: 100% !important;
            text-align: center !important;
            padding: 12px 8px !important;
            transition: all 0.2s ease !important;
        }
        section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
            background: rgba(155,89,245,0.14) !important;
            border-color: rgba(155,89,245,0.55) !important;
            transform: translateY(-1px) !important;
        }

        /* Bouton actif (page courante) */
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background: rgba(155,89,245,0.20) !important;
            border: 1px solid rgba(155,89,245,0.80) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            width: 100% !important;
            text-align: center !important;
            padding: 12px 8px !important;
            box-shadow: 0 0 14px rgba(155,89,245,0.22) !important;
        }

        /* ─── TITRES H1/H2/H3 ─── */
        h1 { font-family: var(--font-title) !important; color: inherit !important; }
        h2 { font-family: var(--font-title) !important; color: inherit !important; }
        h3 { font-family: var(--font-body) !important; font-weight: 700 !important; color: inherit !important; }

        /* Force labels to inherit from text color */
        label, .stSelectbox p {
            color: inherit !important;
        }

        /* ─── BOUTONS GÉNÉRAUX ─── */
        .stButton > button {
            background: rgba(155,89,245,0.10) !important;
            color: #ffffff !important;
            border: 1px solid rgba(155,89,245,0.28) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 10px 24px !important;
            transition: all .2s ease !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.18) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
        }

        .stButton > button:hover {
            background: rgba(155,89,245,0.16) !important;
            border-color: rgba(155,89,245,0.45) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 18px rgba(155,89,245,0.18) !important;
            filter: none !important;
        }

        .stButton > button[kind="primary"] {
            background: rgba(155,89,245,0.18) !important;
            color: #ffffff !important;
            border: 1px solid rgba(155,89,245,0.65) !important;
            box-shadow: 0 0 14px rgba(155,89,245,0.20) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: rgba(155,89,245,0.24) !important;
            border-color: rgba(155,89,245,0.80) !important;
            box-shadow: 0 0 18px rgba(155,89,245,0.28) !important;
        }

        /* ─── INPUTS ─── */
        .stTextInput input, div[data-baseweb="input"] input {
            background: rgba(128,128,128,0.1) !important;
            border: 1px solid var(--border-dim) !important;
            border-radius: 8px !important;
            color: inherit !important;
        }
        .stTextInput input:focus, div[data-baseweb="input"] input:focus {
            border-color: var(--violet) !important;
            box-shadow: 0 0 0 2px var(--violet-glow) !important;
        }
        span[data-baseweb="tag"] {
            background: rgba(155,89,245,0.18) !important;
            color: var(--violet) !important;
            border: 1px solid rgba(155,89,245,0.35) !important;
        }

        /* ─── CONTAINERS BORDER ─── */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface) !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid var(--border-dim) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow) !important;
            transition: border-color .2s, box-shadow .2s;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(155,89,245,0.35) !important;
            box-shadow: 0 4px 24px rgba(155,89,245,0.15) !important;
        }

        /* ─── EXPANDERS ─── */
        details[data-testid="stExpander"] {
            background: var(--surface) !important;
            border: 1px solid var(--border-dim) !important;
            border-radius: var(--radius) !important;
            backdrop-filter: blur(6px);
        }
        details[data-testid="stExpander"] summary {
            font-weight: 600; color: #ccd0e0;
        }

        /* ─── DATAFRAME ─── */
        .stDataFrame { border-radius: var(--radius) !important; overflow: hidden; }

        /* ─── TABS ─── */
        .stTabs [data-baseweb="tab"] {
            background: var(--surface) !important;
            border-radius: 8px 8px 0 0 !important;
            color: #aab !important;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(155,89,245,0.12) !important;
            color: var(--violet) !important;
            border-bottom: 2px solid var(--violet) !important;
        }

        /* ─── METRIC CARDS ─── */
        div[data-testid="stMetric"] {
            background: rgba(155,89,245,0.07);
            border: 1px solid rgba(155,89,245,0.20);
            border-radius: var(--radius);
            padding: 12px 16px;
        }
        div[data-testid="stMetricValue"] {
            color: var(--gold) !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        /* ─── DIVIDER ─── */
        hr {
            border: none;
            border-top: 1px solid rgba(155,89,245,0.20) !important;
            margin: 18px 0 !important;
        }

        /* ─── SCROLLBAR ─── */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(155,89,245,0.35);
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def set_background(image_path: Path, overlay_opacity: float = 0.82):
    """
    Fond image sur toute l'app (.stApp).
    La sidebar est opaque (fond propre) donc elle couvre l'image derrière elle.
    Fallback gradient si image absente.
    """
    FALLBACK = """
    <style>
    .stApp {
        background: linear-gradient(135deg,#0c0b28 0%,#14133a 50%,#0f0e2a 100%) !important;
        background-attachment: fixed !important;
    }
    </style>"""

    if not image_path.exists():
        st.markdown(FALLBACK, unsafe_allow_html=True)
        return

    try:
        b64 = _img_to_base64(image_path)
        ext = image_path.suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg","jpeg") else ("image/webp" if ext=="webp" else "image/png")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background:
                    linear-gradient(rgba(8,8,30,{overlay_opacity}), rgba(10,10,40,{overlay_opacity})),
                    url("data:{mime};base64,{b64}");
                background-size: cover;
                background-position: center top;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* Le sélecteur Sidebar opaque a été supprimé pour laisser agir la transparence. */
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(FALLBACK, unsafe_allow_html=True)


# Alias de compatibilité avec l'ancien code
def set_page_background(image_path: Path, opacity: float = 0.18):
    set_background(image_path, overlay_opacity=0.82)


def render_banner(title: str, subtitle: str = "", icon: str = ""):
    """Banderole premium en haut de chaque sous-page."""
    icon_html = f"<span style='font-size:2rem;margin-right:12px;vertical-align:middle;'>{icon}</span>" if icon else ""
    sub_html = f"<p style='margin:8px 0 0;font-size:1.05rem;color:#aab0cc;font-weight:400;letter-spacing:.04em;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            background: linear-gradient(135deg,rgba(123,47,190,.22) 0%,rgba(61,90,254,.18) 100%);
            border: 1px solid rgba(255,215,0,.18);
            border-radius: 16px;
            padding: 28px 36px;
            margin-bottom: 28px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 4px 32px rgba(0,0,0,.4);
        ">
            <h1 style="
                font-family:'Cinzel',serif;
                font-size:2.6rem;
                font-weight:900;
                color:#ffffff;
                margin:0;
                text-shadow:0 2px 16px rgba(0,0,0,.8), 0 0 32px rgba(255,255,255,.08);
                letter-spacing:.04em;
            ">{icon_html}{title}</h1>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, icon: str = "", accent_color: str = "#FFD700") -> None:
    """Encart glassmorphism réutilisable."""
    st.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,0.05);
            border:1px solid rgba(255,255,255,0.10);
            border-left:4px solid {accent_color};
            border-radius:14px;
            padding:22px 24px;
            margin-bottom:14px;
            backdrop-filter:blur(8px);
            -webkit-backdrop-filter:blur(8px);
            box-shadow:0 6px 28px rgba(0,0,0,.45);
            transition:border-color .2s, box-shadow .2s;
        ">
            <h3 style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;
                       color:{accent_color};margin:0 0 8px;">
                {icon} {title}
            </h3>
            <p style="color:#c0c8e0;font-size:.95rem;margin:0;line-height:1.6;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_dataset_card(df: pd.DataFrame) -> None:
    """Carte stats dataset dans la sidebar (style glassmorphism premium)."""
    if df.empty:
        return

    total_films = len(df)

    if "Genres" in df.columns:
        nb_genres = (
            df["Genres"].fillna("").astype(str)
            .str.split(",").explode().str.strip()
        )
        nb_genres = nb_genres[(nb_genres != "") & (nb_genres != r"\N")].nunique()
    else:
        nb_genres = 0

    if "Annee_de_sortie" in df.columns:
        years = pd.to_datetime(df["Annee_de_sortie"], errors="coerce").dt.year
        if years.dropna().empty:
            periode = "—"
        else:
            periode = f"{int(years.min())} → {int(years.max())}"
    else:
        periode = "—"

    note_moy = f"{df['Note_moyenne'].mean():.1f}" if "Note_moyenne" in df.columns else "—"

    html = (
        '<div style="'
        'background:transparent;'
        'border:none;'
        'padding:10px 0;margin:6px 0 12px;">'

        '<p style="color:#ffffff;font-family:\'Cinzel\',serif;font-size:.75rem;'
        'font-weight:700;letter-spacing:.12em;text-transform:uppercase;'
        'margin:0 0 16px;text-align:center;">STATISTIQUES de la base de données</p>'

        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:center;">'

        # Films
        '<div style="background:rgba(255,255,255,.05);padding:14px 10px;border-radius:12px;'
        'border:1px solid rgba(155,89,245,0.2);">'
        '<div style="font-size:1.4rem;margin-bottom:6px;">&#127909;</div>'
        f'<div style="color:#FFD700;font-size:1.3rem;font-weight:800;line-height:1;margin-bottom:4px;">{total_films:,}</div>'
        '<div style="color:#6870a0;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;">Films</div>'
        '</div>'

        # Genres
        '<div style="background:rgba(255,255,255,.05);padding:14px 10px;border-radius:12px;'
        'border:1px solid rgba(155,89,245,0.2);">'
        '<div style="font-size:1.4rem;margin-bottom:6px;">&#127991;&#65039;</div>'
        f'<div style="color:#a78bfa;font-size:1.3rem;font-weight:800;line-height:1;margin-bottom:4px;">{nb_genres}</div>'
        '<div style="color:#6870a0;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;">Genres</div>'
        '</div>'

        # Période
        '<div style="background:rgba(255,255,255,.05);padding:14px 10px;border-radius:12px;'
        'border:1px solid rgba(155,89,245,0.2);">'
        '<div style="font-size:1.4rem;margin-bottom:6px;">&#128197;</div>'
        f'<div style="color:#60a5fa;font-size:1rem;font-weight:700;line-height:1.3;margin-bottom:4px;">{periode}</div>'
        '<div style="color:#6870a0;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;">Période</div>'
        '</div>'

        # Note moy.
        '<div style="background:rgba(255,255,255,.05);padding:14px 10px;border-radius:12px;'
        'border:1px solid rgba(155,89,245,0.2);">'
        '<div style="font-size:1.4rem;margin-bottom:6px;">&#11088;</div>'
        f'<div style="color:#34d399;font-size:1.3rem;font-weight:800;line-height:1;margin-bottom:4px;">{note_moy}</div>'
        '<div style="color:#6870a0;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;">Note moy.</div>'
        '</div>'

        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
# --------------------
# RECUPERATION DES POSTERS
# --------------------
def _poster_url(value: str) -> str:
    """
    Colonne 'Poster' = URL complète OU chemin TMDB '/xxx.jpg'
    """
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if not v:
        return ""
    if v.startswith("/"):
        return f"https://image.tmdb.org/t/p/w500{v}"
    return v

# --------------------
# MODEL ML
# --------------------
def prepare_df_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare df : types, normalisation titres, dummies genres."""
    d = df.copy()

    # textes
    for col in ["Titre_fr", "Synopsis", "Acteur_actrice", "Realisateur", "Langue_originale", "Genres"]:
        if col in d.columns:
            d[col] = d[col].fillna("").astype(str)
        else:
            d[col] = ""

    # normalisation titre
    d["Titre_fr_norm"] = d["Titre_fr"].str.lower().str.strip()

    # dummies genres
    genres_dummies = d["Genres"].str.get_dummies(sep=",")
    d = pd.concat([genres_dummies, d], axis=1)

    # année -> int
    if "Annee_de_sortie" in d.columns:
        d["Annee_de_sortie"] = pd.to_datetime(d["Annee_de_sortie"], errors="coerce").dt.year
    else:
        d["Annee_de_sortie"] = np.nan

    # numériques
    for col in ["Duree(min)", "Note_moyenne", "Nombre_de_votes", "Budget", "Box_office"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = np.nan

    return d

#--------------------
# FONCTIONS RECOMMANDATION ML
#--------------------
@st.cache_resource(show_spinner=True)
def fit_reco_model(
    df_prepared: pd.DataFrame,
    poids_box: float,
    poids_genre: float,
    poids_acteurs: float,
    poids_real: float,
    poids_note: float,
):
    """
    Entraîne pipeline + matrice X + KNN.
    Cache => recalcul uniquement si df_prepared (CSV) ou les poids changent.
    """

    # Colonnes non-genre pour détecter les dummies genres
    non_genre = {
        "level_0", "index", "ID_film", "Realisateur", "Acteur_actrice", "Titre_original",
        "Titre_original_2", "Annee_de_sortie", "Duree(min)", "Genres", "Titre_fr",
        "Traduction_regionale", "Note_moyenne", "Nombre_de_votes", "Backdrop_path",
        "Budget", "imdb_id", "Langue_originale", "Synopsis", "Poster",
        "Pays_de_production", "Box_office", "Bande_annonce", "Titre_fr_norm"
    }

    # heuristique : colonnes binaires 0/1 => genres
    genre_cols = [
        c for c in df_prepared.columns
        if c not in non_genre and df_prepared[c].dropna().isin([0, 1]).all()
    ]

    col_num = ["Annee_de_sortie", "Duree(min)", "Nombre_de_votes", "Budget"]
    col_box = ["Box_office"]
    col_note = ["Note_moyenne"]
    col_cat = ["Langue_originale"]

    def weight_mult(w: float):
        return FunctionTransformer(lambda X: X * w)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), col_num),
            ("box", make_pipeline(StandardScaler(), weight_mult(poids_box)), col_box),
            ("note", make_pipeline(StandardScaler(), weight_mult(poids_note)), col_note),
            ("cat", OneHotEncoder(handle_unknown="ignore"), col_cat),

            ("genres", weight_mult(poids_genre), genre_cols),

            ("synopsis", TfidfVectorizer(max_features=5000, stop_words="english"), "Synopsis"),

            ("acteurs", make_pipeline(
                CountVectorizer(token_pattern=r"[^,]+"),
                weight_mult(poids_acteurs)
            ), "Acteur_actrice"),

            ("real", make_pipeline(
                CountVectorizer(token_pattern=r"[^,]+"),
                weight_mult(poids_real)
            ), "Realisateur"),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    pipeline = Pipeline(steps=[("preprocessing", preprocessor)])
    X = pipeline.fit_transform(df_prepared)

    model = NearestNeighbors(n_neighbors=6, metric="cosine")
    model.fit(X)

    return X, model


def get_recommendations_ml(
    df: pd.DataFrame,
    df_prepared: pd.DataFrame,
    X,
    model,
    titre_fr: str,
    k: int = 5
) -> pd.DataFrame:
    """Retourne un DF de recommandations + distance cosine."""
    titre_norm = (titre_fr or "").lower().strip()
    if not titre_norm:
        return df.iloc[0:0]

    idx_list = df_prepared.index[df_prepared["Titre_fr_norm"] == titre_norm].tolist()
    if not idx_list:
        return df.iloc[0:0]

    idx = idx_list[0]
    distances, indices = model.kneighbors(X[idx], n_neighbors=k + 1)

    reco_idx = indices[0][1:]  # exclure le film lui-même
    reco = df.loc[reco_idx].copy()
    reco["distance_cosine"] = distances[0][1:]
    return reco

# --------------------
# FONCTIONS DÉTAIL FILM
# --------------------
def render_movie_detail(row: pd.Series):
    """Fiche film détaillée (Style Premium)."""

    # ── HEADER : Titre & Tagline
    titre = row.get("Titre_fr", "Titre inconnu")
    st.markdown(f"""
        <div style="margin-bottom: 24px;">
            <h1 style="font-family:'Cinzel',serif; font-weight:900; margin:0; color:white; font-size:2.5rem;">{titre}</h1>
            <div style="font-style:italic; color:#9B59F5; background:rgba(155,89,245,0.1); border-left:4px solid #9B59F5; padding:8px 16px; margin-top:12px; border-radius:4px;">
                "L'expérience cinématographique transcendée."
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.2], gap="large")

    with c1:
        url = _poster_url(row.get("Poster", ""))
        if url:
            st.markdown(f"""
                <div style="box-shadow: 0 20px 50px rgba(0,0,0,0.8); border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.1);">
                    <img src="{url}" style="width:100%; display:block;">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Pas d’affiche")

        # Petites stats sous le poster
        st.markdown("<br/>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            annee = row.get("Annee_de_sortie")
            annee_val = str(int(pd.to_datetime(annee, errors='coerce').year)) if pd.notna(annee) else "—"
            st.markdown(f"""<div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                <p style="margin:0; font-size:0.75rem; color:#8890cc; text-transform:uppercase; letter-spacing:1px;">Année</p>
                <p style="margin:0; font-size:1.2rem; font-weight:700;">{annee_val}</p>
            </div>""", unsafe_allow_html=True)

            duree = row.get("Duree(min)", np.nan)
            st.markdown(f"""<div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.1); text-align:center; margin-top:12px;">
                <p style="margin:0; font-size:0.75rem; color:#8890cc; text-transform:uppercase; letter-spacing:1px;">Durée</p>
                <p style="margin:0; font-size:1.2rem; font-weight:700;">{int(duree) if pd.notna(duree) else '—'} min</p>
            </div>""", unsafe_allow_html=True)

        with s2:
            note = row.get("Note_moyenne", np.nan)
            st.markdown(f"""<div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                <p style="margin:0; font-size:0.75rem; color:#8890cc; text-transform:uppercase; letter-spacing:1px;">Note</p>
                <p style="margin:0; font-size:1.2rem; font-weight:700; color:#FFD700;">{note if pd.notna(note) else '—'}/10</p>
            </div>""", unsafe_allow_html=True)

            votes = row.get("Nombre_de_votes", np.nan)
            v_val = f"{int(votes):,}".replace(",", " ") if pd.notna(votes) else "—"
            st.markdown(f"""<div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.1); text-align:center; margin-top:12px;">
                <p style="margin:0; font-size:0.75rem; color:#8890cc; text-transform:uppercase; letter-spacing:1px;">Votes</p>
                <p style="margin:0; font-size:1.2rem; font-weight:700;">{v_val}</p>
            </div>""", unsafe_allow_html=True)

    with c2:
        # Bloc "Détails" style GRID (comme sur l'image)
        st.markdown("<p style='font-family:\"Cinzel\",serif; color:#9B59F5; font-size:0.9rem; letter-spacing:2px; margin-bottom:12px;'>✦ DÉTAILS</p>", unsafe_allow_html=True)

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown(f"**🎭 Genres :** {row.get('Genres', '—')}")
            st.markdown(f"**🎬 Réalisation :** {row.get('Realisateur', '—')}")
            # Nettoyage pays : supprimer [ ] ' "
            pays_raw = str(row.get('Pays_de_production', '—'))
            pays_clean = re.sub(r"[\[\]' \"]", "", pays_raw).replace(",", ", ")
            st.markdown(f"**🌍 Pays :** {pays_clean if pays_clean else '—'}")
        with detail_cols[1]:
            st.markdown(f"**🎥 Production :** IMDb Data")
            st.markdown(f"**🗣️ Langue :** {row.get('Langue_originale', '—')}")
            box = row.get("Box_office", 0)
            if pd.isna(box) or box == 0:
                box_str = "non renseigné"
            else:
                box_str = f"{int(box):,}".replace(",", " ")
            st.markdown(f"**💰 Box-office :** {box_str}")

        st.divider()

        # Synopsis
        st.markdown("<p style='font-family:\"Cinzel\",serif; color:#9B59F5; font-size:0.9rem; letter-spacing:2px; margin-bottom:12px;'>✦ RÉSUMÉ</p>", unsafe_allow_html=True)
        synopsis = row.get("Synopsis", "")
        if isinstance(synopsis, str) and synopsis.strip():
            st.markdown(f"<p style='color:#cbd0f0; line-height:1.6;'>{synopsis}</p>", unsafe_allow_html=True)
        else:
            st.caption("Synopsis non renseigné")

        st.divider()

        # Casting
        st.markdown("<p style='font-family:\"Cinzel\",serif; color:#9B59F5; font-size:0.9rem; letter-spacing:2px; margin-bottom:12px;'>✦ CASTING</p>", unsafe_allow_html=True)
        acteurs = row.get("Acteur_actrice", "")
        if isinstance(acteurs, str) and acteurs.strip():
            st.markdown(f"<div style='background:rgba(255,255,255,0.03); padding:12px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);'>{acteurs}</div>", unsafe_allow_html=True)


def render_movie_card(row: pd.Series, df: pd.DataFrame, df_prepared: pd.DataFrame, X, model, key_prefix: str = "card"):
    """Composant universel pour afficher un film en grille avec bouton 'oeil'."""
    titre = row.get("Titre_fr", "Inconnu")
    poster_link = _poster_url(row.get("Poster", ""))
    if not poster_link:
        poster_link = "https://via.placeholder.com/500x750/1a1a2e/ffffff?text=Pas+d'affiche"

    annee = "—"
    if "Annee_de_sortie" in row and pd.notna(row["Annee_de_sortie"]):
        try: annee = str(int(pd.to_datetime(row["Annee_de_sortie"]).year))
        except: pass

    note = f"{row.get('Note_moyenne', 0):.1f}" if pd.notna(row.get("Note_moyenne")) else "—"
    votes = "—"
    if "Nombre_de_votes" in row and pd.notna(row["Nombre_de_votes"]):
        votes = f"{int(row['Nombre_de_votes']):,}".replace(",", " ")

    st.markdown(f"""
    <div class="movie-card" title="{titre}">
        <div class="movie-poster-container">
            <img src="{poster_link}" alt="{titre}" loading="lazy">
            <div class="movie-gradient"></div>
            <div class="movie-info">
                <div class="movie-title">{titre}</div>
                <div class="movie-meta">
                    <span>{annee}</span>
                    <span style="color:#FFD700;">★ {note}</span>
                </div>
                <div class="movie-votes"><span>👥</span> {votes}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton Streamlit overlay invisible pour le clic
    if st.button("Voir la fiche", key=f"btn_{key_prefix}_{row.name}", use_container_width=True):
        st.session_state["modal_active_movie"] = titre
        st.rerun()

# --------------------
# PAGES ACCEUIL
# --------------------
def page_accueil(df: pd.DataFrame):
    # ── Hero section ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:70px 24px 40px;">
            <p style="font-family:'Cinzel',serif;font-size:.85rem;letter-spacing:.4em;
                      color:#9B59F5;text-transform:uppercase;margin:0 0 18px;">
                ✦ &nbsp;ANALYSE &amp; RECOMMANDATION&nbsp; ✦
            </p>
            <h1 style="font-family:'Cinzel',serif;
                       font-size:clamp(3rem,8vw,5.5rem);
                       font-weight:900;margin:0;line-height:1.0;
                       text-shadow:0 0 50px rgba(155,89,245,.65), 0 0 100px rgba(155,89,245,.25), 0 4px 20px rgba(0,0,0,.95);">
                <span style="color:#FFD700;">CINÉ</span><span style="color:#ffffff;">VISION</span>
            </h1>
            <p style="font-size:1.1rem;color:#8890b0;margin:22px 0 0;
                      letter-spacing:.1em;font-weight:400;">
                Explore &nbsp;&bull;&nbsp; Analyse &nbsp;&bull;&nbsp; Découvre
            </p>
            <div style="width:120px;height:2px;margin:26px auto 0;
                        background:linear-gradient(90deg,transparent,#9B59F5,transparent);
                        border-radius:2px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error(f"CSV introuvable. Mets `{DATA_PATH}` dans le même dossier que le script.")
        return

    # ── 3 feature cards premium ───────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="large")

    _card_base = """
        background:rgba(255,255,255,0.04);
        border-radius:20px;
        padding:36px 24px 30px;
        backdrop-filter:blur(12px);
        -webkit-backdrop-filter:blur(12px);
        box-shadow:0 10px 40px rgba(0,0,0,.55);
        text-align:center;
        min-height:220px;
        transition: transform .2s, box-shadow .2s;
    """

    with col1:
        st.markdown(f"""
        <div style="{_card_base}border:1px solid rgba(96,165,250,0.30);">
            <div style="font-size:3rem;margin-bottom:16px;">&#128269;</div>
            <h3 style="font-family:'Cinzel',serif;font-size:1rem;font-weight:700;
                       color:#60a5fa;margin:0 0 12px;letter-spacing:.08em;">
                RECHERCHE
            </h3>
            <p style="color:#7880a0;font-size:.875rem;line-height:1.7;margin:0;">
                Filtre par <b style='color:#b0beee;'>acteur</b>,
                <b style='color:#b0beee;'>réalisateur</b>,
                <b style='color:#b0beee;'>genre</b> ou <b style='color:#b0beee;'>année</b>.
                Accède à la fiche complète de chaque film.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{_card_base}border:1px solid rgba(255,215,0,0.30);">
            <div style="font-size:3rem;margin-bottom:16px;">&#128202;</div>
            <h3 style="font-family:'Cinzel',serif;font-size:1rem;font-weight:700;
                       color:#FFD700;margin:0 0 12px;letter-spacing:.08em;">
                KPIs &amp; ANALYSES
            </h3>
            <p style="color:#7880a0;font-size:.875rem;line-height:1.7;margin:0;">
                Tendances du cinéma : <b style='color:#b0beee;'>notes</b>,
                <b style='color:#b0beee;'>revenus</b>, <b style='color:#b0beee;'>acteurs</b>
                et évolution par décennie.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="{_card_base}border:1px solid rgba(155,89,245,0.40);">
            <div style="font-size:3rem;margin-bottom:16px;">&#129302;</div>
            <h3 style="font-family:'Cinzel',serif;font-size:1rem;font-weight:700;
                       color:#a78bfa;margin:0 0 12px;letter-spacing:.08em;">
                RECOMMANDATION ML
            </h3>
            <p style="color:#7880a0;font-size:.875rem;line-height:1.7;margin:0;">
                Moteur de recommandation : <b style='color:#b0beee;'>5 films similaires</b>
                 fondés sur tes critères favoris.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# --------------------
# PAGES RECHERCHE
# --------------------
def page_recherche(df: pd.DataFrame, filtres: dict):
    render_banner(
        title="Catalogue & Recherche",
        subtitle="Filtre rapidement les films par acteur, réalisateur, genre…",
        icon="",
    )

    if df.empty:
        st.error("CSV introuvable.")
        return

    # Récupération des filtres depuis la sidebar
    title_q       = filtres.get("title_q", "")
    use_actor     = filtres.get("use_actor", False)
    actor_q       = filtres.get("actor_q", "")
    use_real      = filtres.get("use_real", False)
    real_q        = filtres.get("real_q", "")
    use_genre     = filtres.get("use_genre", False)
    selected_genres = filtres.get("selected_genres", [])
    use_year      = filtres.get("use_year", False)
    year_min      = filtres.get("year_min", None)
    year_max      = filtres.get("year_max", None)
    use_lang      = filtres.get("use_lang", False)
    selected_lang = filtres.get("selected_lang", [])

    # --- application filtres
    d = df.copy()

    if title_q.strip():
        q = re.escape(title_q.strip())
        d = d[d["Titre_fr"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=True)]

    if use_actor and actor_q.strip():
        q = re.escape(actor_q.strip())
        d = d[d["Acteur_actrice"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=True)]

    if use_real and real_q.strip():
        q = re.escape(real_q.strip())
        d = d[d["Realisateur"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=True)]

    if use_genre and selected_genres:
        pattern = "|".join([rf"(^|,\s*){re.escape(g)}(,|$)" for g in selected_genres])
        d = d[d["Genres"].fillna("").astype(str).str.contains(pattern, case=False, na=False, regex=True)]

    if use_year and year_min is not None and "Annee_de_sortie" in d.columns:
        y = pd.to_datetime(d["Annee_de_sortie"], errors="coerce").dt.year
        d = d[(y >= year_min) & (y <= year_max)]

    if use_lang and selected_lang and "Langue_originale" in d.columns:
        d = d[d["Langue_originale"].fillna("").astype(str).isin(selected_lang)]

    # On s'assure que le modèle ML est prêt pour les modals
    df_prepared = prepare_df_for_ml(df)
    X, model = fit_reco_model(
        df_prepared=df_prepared,
        poids_box=1.0,
        poids_genre=1.0,
        poids_acteurs=1.0,
        poids_real=1.0,
        poids_note=1.0,
    )

    st.divider()

    if d.empty:
        st.info("ℹ️ Aucun résultat. Ajuste les filtres dans la barre latérale.")
        return

    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

    st.subheader(f"🎯 Explorer notre catalogue : {len(d)} film(s)")

    # Mode tableau détaillé en dessous avec un expander
    with st.expander("Voir les résultats au format tableau de données"):
        cols_show = [c for c in ["Titre_fr", "Annee_de_sortie", "Genres", "Note_moyenne", "Nombre_de_votes", "Realisateur", "Box_office"] if c in d.columns]
        # Formattage dataframe
        st.dataframe(
            d[cols_show].head(200),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Note_moyenne": st.column_config.NumberColumn("Note", format="%.1f ⭐"),
                "Nombre_de_votes": st.column_config.NumberColumn("Votes"),
                "Annee_de_sortie": st.column_config.DateColumn("Année", format="YYYY")
            }
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Grille de films stylisée (Netflix style)
    # Limiter à 50 films pour éviter de faire ramer le navigateur
    d_view = d.head(50)

    # CSS pour le hover des posters
    st.markdown("""
    <style>
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 24px;
        cursor: pointer;
        position: relative;
    }
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(155, 89, 245, 0.4);
        border-color: rgba(155, 89, 245, 0.5);
    }
    .movie-poster-container {
        width: 100%;
        aspect-ratio: 2/3;
        overflow: hidden;
        position: relative;
    }
    .movie-poster-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .movie-gradient {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 80%;
        background: linear-gradient(to top, rgba(10,10,25,0.95) 0%, rgba(10,10,25,0.6) 40%, transparent 100%);
    }
    .movie-info {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 16px;
    }
    .movie-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        color: white;
        margin-bottom: 6px;
        line-height: 1.2;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-meta {
        font-size: 0.8rem;
        color: #ccd0e0;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
    }
    .movie-votes {
        font-size: 0.75rem;
        color: #8890cc;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(5)  # 5 films par ligne

    for i, (_, row) in enumerate(d_view.iterrows()):
        col = cols[i % 5]
        with col:
            render_movie_card(row, df, df_prepared, X, model, key_prefix="cat")

    if len(d) > 50:
        st.markdown(f"<p style='text-align:center;color:#8890cc;margin-top:20px;'>(50 premiers résultats affichés sur {len(d)})</p>", unsafe_allow_html=True)

    st.divider()

# ==============================================================
# PAGE KPIs
# ==============================================================
def page_kpis(df: pd.DataFrame):
    render_banner(
        title="Statistiques et analyses des films",
        subtitle="Analyse des données pour mieux comprendre les tendances du cinéma et les préférences du public.",
        icon="",
    )
    plt.style.use("dark_background")
    # Forcer la transparence du fond des graphiques
    plt.rcParams.update({
        "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0)
    })
    if df.empty:
        st.error("CSV introuvable.")
        return

    # PRÉPARATION DES DONNÉES
    df_copy = df.copy()
    df_copy['Genres'] = df_copy['Genres'].str.split(',')
    df_exploded = df_copy.explode('Genres')
    df_exploded['Genres'] = df_exploded['Genres'].str.strip()

    dfAntonio = df.copy()
    Langue = {
        "fr": "Français", "en": "Américains", "de": "Allemand",
        "es": "Espagnol", "it": "Italien", "pt": "Portugais",
        "nl": "Néerlandais", "da": "Danois", "no": "Norvégien"
    }
    dfAntonio["Langue_originale"] = dfAntonio["Langue_originale"].replace(Langue)
    dfAntonio['Annee_de_sortie'] = pd.to_datetime(dfAntonio['Annee_de_sortie'], errors='coerce')
    dfAntonio["Année"] = dfAntonio['Annee_de_sortie'].dt.year

    # LISTE DÉROULANTE
    kpi_choix = st.selectbox("Choisir un KPI :", [
        "Catégories de durée et durée moyenne par genre",
        "Note moyenne des films par genre",
        "Top 10 des films générant le plus de revenus",
        "Top 10 acteurs/actrices les plus présents",
        "Top 10 réalisateurs les plus présents",
        "Revenue moyen pour un film français par décennie",
        "Évolution du nombre de films français par décennie",
    ])


    # ==============================================================
    # KPI 1 — Subplot : Note par catégorie de durée + Durée moyenne par genre
    # ==============================================================
    if kpi_choix == "Catégories de durée et durée moyenne par genre":
        # Données catégorie durée
        df_temp = df.copy()
        bins = [0, 90, 120, 150, 1000]
        labels = ['Court (<90min)', 'Moyen (90-120min)', 'Long (120-150min)', 'Très long (>150min)']
        df_temp['Categorie_duree'] = pd.cut(df_temp['Duree(min)'], bins=bins, labels=labels)
        note_par_cat = (df_temp.groupby('Categorie_duree', observed=True)['Note_moyenne']
            .mean().sort_values(ascending=False).reset_index().round(2))

        # Données durée moyenne par genre
        Temps_moyen = (df_exploded.groupby(['Genres'])['Duree(min)']
            .mean().sort_values(ascending=False).head(10).reset_index().round(2))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Subplot gauche
        ax1.barh(note_par_cat['Categorie_duree'], note_par_cat['Note_moyenne'],
                 color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c'])
        for i, valeur in enumerate(note_par_cat['Note_moyenne']):
            ax1.text(valeur + 0.05, i, str(valeur), va='center', fontweight='bold')
        ax1.set_xlabel('Note moyenne')
        ax1.set_title('Note moyenne par catégorie de durée', fontsize=13, fontweight='bold')
        ax1.set_xlim(0, 10)

        # Subplot droit
        ax2.barh(Temps_moyen['Genres'], Temps_moyen['Duree(min)'], color='steelblue')
        for i, valeur in enumerate(Temps_moyen['Duree(min)']):
            ax2.text(valeur + 0.5, i, f'{valeur:.1f} min', va='center', fontweight='bold')
        ax2.set_xlabel('Durée moyenne (min)')
        ax2.set_title('Durée moyenne des films par genre', fontsize=13, fontweight='bold')
        ax2.set_xlim(0, Temps_moyen['Duree(min)'].max() * 1.2)

        plt.suptitle('Analyse des durées', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI analyse la durée des films sous deux angles.\n\n"
        "1) **Note moyenne par catégorie de durée** : on observe une tendance globale où les films plus longs "
        "obtiennent une meilleure note moyenne (les films très longs ressortent en tête).\n\n"
        "2) **Durée moyenne par genre** : certains genres comme **History / Biography / War** ont des durées "
        "moyennes plus élevées, tandis que d'autres restent plus proches du format standard.\n\n"
        "**Objectif** : comprendre l’impact de la durée sur la perception des films et identifier les genres "
        "associés à des durées plus longues ou plus courtes.\n\n")
        st.divider()

    # ==============================================================
    # KPI 2 — Note moyenne des films par genre
    # ==============================================================
    elif kpi_choix == "Note moyenne des films par genre":
        top10_genres = (df_exploded[df_exploded['Genres'] != '\\\\N']
            .groupby('Genres')['Note_moyenne'].mean()
            .sort_values(ascending=False).head(10).reset_index())
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top10_genres['Genres'], top10_genres['Note_moyenne'], color='steelblue')
        for i, valeur in enumerate(top10_genres['Note_moyenne']):
            ax.text(valeur + 0.05, i, str(round(valeur, 2)), va='center', fontweight='bold')
        ax.set_xlabel('Note moyenne')
        ax.set_title('Note moyenne des films par genre', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI présente la note moyenne des films par genre.\n\n"
        "Les genres **Documentary, Biography et History** ressortent comme les mieux notés, "
        "tandis que des genres comme **Mystery ou Adventure** obtiennent des moyennes légèrement inférieures.\n\n"
        "L'écart global reste modéré (~1 point), ce qui suggère une certaine homogénéité "
        "des notes dans la base.\n\n"
        "**Objectif** : identifier les genres perçus comme les plus qualitatifs et "
        "comprendre l'impact des genres possibles dans les recommandations.\n\n")
        st.divider()

    # ==============================================================
    # KPI 3 — Top 10 films générant le plus de revenus
    # ==============================================================
    elif kpi_choix == "Top 10 des films générant le plus de revenus":
        Top10revenue = (df[['Titre_fr', 'Box_office']]
            .sort_values(by='Box_office', ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(Top10revenue['Titre_fr'], Top10revenue['Box_office'], color='steelblue')
        for i, valeur in enumerate(Top10revenue['Box_office']):
            ax.text(valeur + 10000000, i, f'{valeur/1e9:.2f}Md$', va='center', fontweight='bold')
        ax.set_xlabel('Box office ($)')
        ax.set_title('Top 10 des films générant le plus de revenus', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 3.5e9)
        ax.xaxis.get_offset_text().set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI met en avant les films les plus rentables de la base.\n\n"
        "On retrouve en tête : **Avatar** (~2.92 Md$) , **Avengers: Endgame**,  et **Avatar : La Voie de l’eau**.\n\n"
        "Le classement est très marqué par les **franchises** "
        "(Marvel, Star Wars, …), ce qui montre une forte présence de blockbusters.\n\n" \
        "**Objectif** : identifier les titres les plus rentables et repérer"
        "les films très populaires dans le dataset.\n\n")
        st.divider()

    # ==============================================================
    # KPI 4 — Top 10 acteurs/actrices les plus présents
    # ==============================================================
    elif kpi_choix == "Top 10 acteurs/actrices les plus présents":
        df_Nico_copy = df.copy()
        df_Nico_copy['Acteur_actrice'] = df_Nico_copy['Acteur_actrice'].fillna("").str.split(',')
        df_Nico_exploded = df_Nico_copy.explode('Acteur_actrice')
        df_Nico_exploded['Acteur_actrice'] = df_Nico_exploded['Acteur_actrice'].str.strip()
        df_Nico_exploded = df_Nico_exploded[df_Nico_exploded['Acteur_actrice'] != ""]
        fig, ax = plt.subplots(figsize=(14, 8))
        df_Nico_exploded['Acteur_actrice'].value_counts().head(10).plot(kind="bar", ax=ax, color="gold")
        ax.set_title("Top 10 des acteurs/actrices les plus présents dans les films")
        ax.set_xlabel("Acteur/Actrice")
        ax.set_ylabel("Nombre de films")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI permet d'identifier les acteurs/actrices les plus fréquents dans le dataset.\n\n"
        "On retrouve en tête : **Tom Hanks**, **Leonardo DiCaprio**, **Brad Pitt** et **Meryl Streep**.\n\n"
        "**Objectif** : identifier les acteurs/actrices les plus présents et repérer "
        "les figures clés du cinéma.\n\n")
        st.divider()

    # ==============================================================
    # KPI 5 — Top 10 réalisateurs les plus présents
    # ==============================================================
    elif kpi_choix == "Top 10 réalisateurs les plus présents":
        df_real = df.copy()
        df_real['Realisateur'] = (df_real['Realisateur'].fillna("Non renseigné")
            .replace(r"^\s*$", "Non renseigné", regex=True))
        df_real['Realisateur'] = df_real['Realisateur'].str.split(',')
        df_real_exploded = df_real.explode('Realisateur')
        df_real_exploded['Realisateur'] = df_real_exploded['Realisateur'].str.strip()
        df_real_exploded = df_real_exploded[df_real_exploded['Realisateur'] != "Non renseigné"]
        fig, ax = plt.subplots(figsize=(14, 8))
        df_real_exploded['Realisateur'].value_counts().head(10).plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Top 10 des réalisateurs ayant fait le plus de films")
        ax.set_xlabel("Réalisateur")
        ax.set_ylabel("Nombre de films")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI permet d'identifier les réalisateurs les plus fréquents dans le dataset.\n\n"
        "On retrouve en tête : **Steven Spielberg**, **Christopher Nolan**, **Quentin Tarantino** et **Martin Scorsese**.\n\n"
        "**Objectif** : identifier les réalisateurs les plus présents et repérer "
        "les figures clés du cinéma.\n\n")
        st.divider()

    # ==============================================================
    # KPI 6 — Revenue moyen pour un film français par décennie
    # ==============================================================
    elif kpi_choix == "Revenue moyen pour un film français par décennie":
        dfAntonioFR = dfAntonio[dfAntonio["Langue_originale"] == "Français"].copy()
        dfAntonioFR = dfAntonioFR[dfAntonioFR['Budget'] >= 0]
        bins1 = [1945, 1955, 1965, 1975, 1985, 1995, 2005, 2015, 2025]
        labels1 = ['1945–1955', '1955–1965', '1965–1975', '1975–1985',
                   '1985–1995', '1995–2005', '2005–2015', '2015–2025']
        dfAntonioFR['Categorie_Année'] = pd.cut(dfAntonioFR["Année"], bins=bins1, labels=labels1)
        box_fr = (dfAntonioFR.groupby(['Categorie_Année', 'Langue_originale'], observed=True)['Box_office']
            .mean().reset_index().round(2))
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='Categorie_Année', y='Box_office', data=box_fr, color="red", ax=ax)
        sns.barplot(x='Categorie_Année', y='Box_office', data=box_fr, ax=ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e3:.0f} K€"))
        for container in ax.containers:
            ax.bar_label(container, color="white", fmt=lambda x: f"{x/1e3:.1f} K€")
        ax.set_xlabel("Décennie")
        ax.set_ylabel("Revenue en milliers $")
        ax.set_title("Revenue moyen pour un film français par décennie")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI permet d'identifier l'évolution du revenue moyen pour un film français par décennie.\n\n"
        "On observe une forte variation de la moyenne de revenue au fil des décennies, "
        "avec des pics notables dans les années 1980 et 1990.\n\n"
        "**Objectif** : identifier les décennies où les films français ont connu le plus fort revenu.\n\n")
        st.divider()

    # ==============================================================
    # KPI 7 — Subplot : Évolution films français + Répartition par décennie
    # ==============================================================
    elif kpi_choix == "Évolution du nombre de films français par décennie":
        # Données films français par décennie
        dfAntonioFR = dfAntonio[dfAntonio["Langue_originale"] == "Français"].copy()
        bins = [1945, 1955, 1965, 1975, 1985, 1995, 2005, 2015, 2025]
        labels = ['1945–1955', '1955–1965', '1965–1975', '1975–1985',
                  '1985–1995', '1995–2005', '2005–2015', '2015–2025']
        dfAntonioFR['Categorie_Année'] = pd.cut(dfAntonioFR["Année"], bins=bins, labels=labels)
        nb_films_fr = (dfAntonioFR.groupby(['Categorie_Année', 'Langue_originale'], observed=True)['Box_office']
            .count().reset_index().round(2))

        # Données répartition globale par année
        df_temp = df.copy()
        df_temp['Annee_de_sortie'] = pd.to_datetime(df_temp['Annee_de_sortie'], errors='coerce').dt.year
        counts = df_temp["Annee_de_sortie"].value_counts().sort_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Subplot gauche — films français par décennie
        sns.lineplot(x='Categorie_Année', y='Box_office', data=nb_films_fr, color="red", ax=ax1)
        sns.barplot(x='Categorie_Année', y='Box_office', data=nb_films_fr, ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, color="white")
        ax1.set_xlabel("Décennie")
        ax1.set_ylabel("Nombre de films")
        ax1.set_title("Évolution du nombre de films français par décennie", fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Subplot droit — répartition globale par année
        counts.plot(kind="bar", ax=ax2)
        ax2.set_xticks(range(0, len(counts), 10))
        ax2.set_xticklabels(counts.index[::10], rotation=45)
        ax2.set_title("Répartition globale des films par année", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Nombre de films")

        plt.suptitle('Évolution de la production cinématographique', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown(
"""
<h2 style='text-align: center;'>Analyse et interprétation</h2>
<p style='text-align: center; font-size:18px;'>
</p>
""",
unsafe_allow_html=True
)
        st.write("Ce KPI permet d'identifier l'évolution du nombre de films français par décennie.\n\n"
        "On observe une forte variation du nombre de films par décennie, "
        "avec des pics notables dans les années 1980 et 1990.\n\n"
        "**Objectif** : identifier les décennies où les films français ont connu le plus fort nombre de productions.\n\n")
        st.divider()

# --------------------
# PAGE SÉLECTION DE FILMS — RECOMMANDATION ML
# --------------------
@st.dialog("🎬 Détails du film", width="large")
def dialog_movie_detail(df: pd.DataFrame, df_prepared: pd.DataFrame, X, model):
    """Affiche le détail d'un film et ses propres recommandations associées dans une pop-up."""
    titre_fr = st.session_state.get("modal_active_movie")
    if not titre_fr:
        return

    try:
        row = df[df["Titre_fr"] == titre_fr].iloc[0]
        render_movie_detail(row)
        st.divider()
        st.subheader("💡 Films similaires (Recommandations secondaires)")

        reco_df = get_recommendations_ml(df, df_prepared, X, model, titre_fr, k=5)
        if reco_df.empty:
            st.info("Aucune recommandation disponible pour ce film.")
        else:
            reco_df = reco_df.sort_values("distance_cosine", ascending=True)
            cols = st.columns(5)
            for i, (_, r) in enumerate(reco_df.iterrows()):
                with cols[i % 5]:
                    render_movie_card(r, df, df_prepared, X, model, key_prefix="modal_reco")

        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("❌ Fermer la fiche", use_container_width=True):
            del st.session_state["modal_active_movie"]
            st.rerun()

    except Exception as e:
        st.error(f"Erreur lors de l'affichage du film : {e}")
        if st.button("Fermer"):
            del st.session_state["modal_active_movie"]
            st.rerun()

def page_selection_films_ml(df: pd.DataFrame, poids: dict):
    render_banner(
        title="Suggestions de films",
        subtitle="Recherche de films + réglage de l'importance des critères dans la recherche de films similaires.",
        icon="",
    )
    if df.empty:
        st.error(f"CSV introuvable. Mettre `{DATA_PATH}` dans le même dossier que le script.")
        return

    # Récupération des poids depuis la sidebar
    w_box   = poids.get("box",   "Moyen")
    w_genre = poids.get("genre", "Moyen")
    w_act   = poids.get("act",   "Moyen")
    w_real  = poids.get("real",  "Moyen")
    w_note  = poids.get("note",  "Moyen")

    query = st.text_input("Recherche de film :", "")
    if query.strip():
        sugg_df = df[df["Titre_fr"].str.lower().str.contains(query.lower().strip(), na=False)].head(30)
    else:
        sugg_df = df.head(30)

    if sugg_df.empty:
        st.warning("Aucun titre trouvé. Essaie un autre mot-clé.")
        return

    selected_title = st.selectbox("Choisis le film :", options=sugg_df["Titre_fr"].tolist(), index=0)

    # Fiche pleine largeur
    selected_row = df[df["Titre_fr"] == selected_title].iloc[0]
    with st.container(border=True):
        render_movie_detail(selected_row)

    df_prepared = prepare_df_for_ml(df)
    X, model = fit_reco_model(
        df_prepared=df_prepared,
        poids_box=POIDS[w_box],
        poids_genre=POIDS[w_genre],
        poids_acteurs=POIDS[w_act],
        poids_real=POIDS[w_real],
        poids_note=POIDS[w_note],
    )

    launch_btn = st.button("🔎 Lancer la recommandation", type="primary", use_container_width=True)

    if launch_btn:
        reco_df = get_recommendations_ml(df, df_prepared, X, model, selected_title, k=5)

        if reco_df.empty:
            st.error("Impossible de générer des recommandations (titre non trouvé après normalisation).")
        else:
            st.session_state["current_reco_results"] = reco_df.sort_values("distance_cosine", ascending=True)

    # Affichage persistant des résultats
    if "current_reco_results" in st.session_state:
        reco_df = st.session_state["current_reco_results"]
        st.subheader("🍿 5 films recommandés")

        cols = st.columns(5)
        for i, (_, r) in enumerate(reco_df.iterrows()):
            with cols[i % 5]:
                render_movie_card(r, df, df_prepared, X, model, key_prefix="reco")

        if st.button("🗑️ Effacer les résultats"):
            if "current_reco_results" in st.session_state:
                del st.session_state["current_reco_results"]
            if "modal_active_movie" in st.session_state:
                del st.session_state["modal_active_movie"]
            st.rerun()

# --------------------
# PAGES A PROPOS
# --------------------
def page_a_propos(df: pd.DataFrame):
    render_banner(
        title="À propos du projet",
        subtitle="Moteur de recommandation de films — Analyse de données cinématographiques.",
        icon="",
    )

    st.divider()

    # OBJECTIFS (pleine largeur)
    st.markdown("<h3 style='text-align:center;'>🎯 Objectifs du projet</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; color:#cbd0f0; margin-bottom:20px;">
            <p>Explorer et analyser les tendances du cinéma pour notre client.</p>
            <p>Construire un moteur de recommandation efficace, basé sur les similarités entre films.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # 2 COLONNES (2 lignes) => raccourcit la page
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='text-align:center;'>📊 Données utilisées</h3>", unsafe_allow_html=True)
            st.markdown(
                """
                Les données proviennent de la base publique **IMDb**.

                Elles incluent :
                - Informations sur les films (titre, année, genres)
                - Notes et popularité
                - Acteurs et réalisateurs
                - Métadonnées textuelles
                """
            )

    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align:center;'>🗓️ Période couverte</h3>", unsafe_allow_html=True)
            st.markdown(
                """
                La base couvre la période **1946 à 2023**.

                Cette amplitude permet :
                - Une analyse des tendances par décennie
                - L’étude de l’évolution des genres
                - L’observation des transformations du marché cinématographique
                """
            )

    col3, col4 = st.columns(2, gap="large")

    with col3:
        with st.container(border=True):
            st.markdown("<h3 style='text-align:center;'>🧠 Méthodologie</h3>", unsafe_allow_html=True)
            st.markdown(
                """
                - Nettoyage et préparation des données
                - Feature engineering
                - Standardisation des variables numériques
                - Vectorisation des données textuelles (TF-IDF)
                - Modèle de similarité avec Nearest Neighbors
                - Évaluation qualitative des recommandations
                - Visualisation des KPIs
                - Itération sur les réglages de poids
                - Documentation et présentation des résultats
                """
            )

    with col4:
        with st.container(border=True):
            st.markdown("<h3 style='text-align:center;'>⚙️ Stack technique</h3>", unsafe_allow_html=True)
            st.markdown(
                """
                - Python
                - Pandas / NumPy
                - Scikit-learn
                - Streamlit
                - Git/GitHub
                - Visual Studio Code
                - Matplotlib / Seaborn
                - Jupyter Notebook
                - Canva (pour les visuels)
                """
            )

    st.divider()

    # ÉQUIPE (centrée)
    st.markdown("<h3 style='text-align:center;margin-bottom:10px;'>👥 Équipe</h3>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])
    with center:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; font-size:16px;">
                    <p style="margin:6px 0;"><b>Antonio BALESTRA</b> — SCRUM MASTER / Data analyst</p>
                    <p style="margin:6px 0;"><b>Nicolas DUWEZ</b> — Data analyst</p>
                    <p style="margin:6px 0;"><b>Frederic DAYAS</b> — Data analyst</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# --------------------
# APP MAIN
# --------------------
def main():
    # ── CSS global + background (toujours en premier) ─────────────────
    inject_global_css()
    set_background(IMG_HOME)

    df = load_data(DATA_PATH)

    if "Annee_de_sortie" in df.columns:
        df["Annee_de_sortie"] = pd.to_datetime(df["Annee_de_sortie"], errors="coerce")

    # ── SIDEBAR PREMIUM ───────────────────────────────────────────────
    with st.sidebar:
        # Logo / titre premium
        st.markdown(
            """
            <div style="text-align:center;padding:18px 4px 10px;">
                <div style="font-family:'Cinzel',serif;font-size:1.6rem;font-weight:900;
                            color:#FFD700;letter-spacing:.1em;line-height:1.1;
                            text-shadow:0 0 16px rgba(255,215,0,.45);">
                    🎬 <span style="color:#FFD700;">CINÉ</span><span style="color:#ffffff;">VISION</span>
                </div>
                <div style="font-size:.72rem;color:#7880a0;letter-spacing:.2em;
                            text-transform:uppercase;margin-top:4px;">
                    Films &amp; Recommandations
                </div>
            </div>
            <hr style="border-color:rgba(255,215,0,.18);margin:4px 0 14px;"/>
            """,
            unsafe_allow_html=True,
        )


        # Navigation : 5 boutons st.button dans une grille 2x2 + 1 pleine largeur
        if "page" not in st.session_state:
            st.session_state["page"] = "Accueil"

        PAGES = ["Accueil", "Catalogue", "Suggestions", "Statistiques"]
        for pg in PAGES:
            btn_type = "primary" if st.session_state["page"] == pg else "secondary"
            if st.button(pg, key=f"nav_{pg}", type=btn_type, use_container_width=True):
                st.session_state["page"] = pg
                if "modal_active_movie" in st.session_state:
                    del st.session_state["modal_active_movie"]
                st.rerun()
        # "À propos" pleine largeur
        btn_type = "primary" if st.session_state["page"] == "À propos" else "secondary"
        if st.button("À propos", key="nav_apropos", type=btn_type, use_container_width=True):
            st.session_state["page"] = "À propos"
            if "modal_active_movie" in st.session_state:
                del st.session_state["modal_active_movie"]
            st.rerun()

        choice = st.session_state["page"]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Carte dataset premium (masquée sur Catalogue et Suggestions pour gagner de la place)
        if not df.empty and choice not in ["Catalogue", "Suggestions"]:
            sidebar_dataset_card(df)
        elif df.empty:
            st.warning(f"CSV introuvable : {DATA_PATH}")

        # Note contexte
        st.markdown(
            """
            <div style="font-size:.78rem;color:#6870a0;line-height:1.55;padding:4px 2px;text-align:center;">
                Films français depuis <b style='color:#a8b0dc;'>1946</b><br>
                Films internationaux majoritairement depuis <b style='color:#a8b0dc;'>2000</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:rgba(255,255,255,.08);margin:14px 0;'/>", unsafe_allow_html=True)

        # ── SIDEBAR CONTEXTUELLE ──────────────────────────────────
        filtres = {}
        poids   = {}

        if choice == "Catalogue":
            genres_uniques = (
                df["Genres"].fillna("").astype(str)
                .str.split(",").explode().str.strip()
            )
            genres_uniques = sorted([g for g in genres_uniques.unique().tolist() if g and g != "\n"])

            st.markdown("""
                <p style='color:#ffffff;font-family:\'Cinzel\',serif;font-size:.75rem;
                font-weight:700;letter-spacing:.12em;text-transform:uppercase;
                margin:0 0 14px;text-align:center;'>FILTRES DU CATALOGUE</p>
            """, unsafe_allow_html=True)

            title_q = st.text_input("🔍 Titre", value="", placeholder="ex: Avatar, Matrix...", key="cat_title")
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:10px 0;'/>", unsafe_allow_html=True)

            use_actor = st.checkbox("Acteur/Actrice", value=False, key="cat_use_actor")
            actor_q = st.text_input("Nom acteur", value="", placeholder="ex: DiCaprio", label_visibility="collapsed", key="cat_actor") if use_actor else ""

            use_real = st.checkbox("Réalisateur", value=False, key="cat_use_real")
            real_q = st.text_input("Nom réalisateur", value="", placeholder="ex: Nolan", label_visibility="collapsed", key="cat_real") if use_real else ""

            use_genre = st.checkbox("Genre", value=False, key="cat_use_genre")
            selected_genres = st.multiselect("Genres", options=genres_uniques, default=[], key="cat_genres") if use_genre else []

            use_year = st.checkbox("Année", value=False, key="cat_use_year")
            year_min, year_max = None, None
            if use_year and "Annee_de_sortie" in df.columns:
                years = pd.to_datetime(df["Annee_de_sortie"], errors="coerce").dt.year
                y_min = int(years.min()) if years.dropna().size else 1900
                y_max = int(years.max()) if years.dropna().size else 2025
                year_min, year_max = st.slider("Époque", min_value=y_min, max_value=y_max, value=(y_min, y_max), key="cat_year")

            use_lang = st.checkbox("Langue", value=False, key="cat_use_lang")
            selected_lang = []
            if use_lang and "Langue_originale" in df.columns:
                langues = sorted([x for x in df["Langue_originale"].dropna().astype(str).unique().tolist() if x.strip()])
                selected_lang = st.multiselect("Langue", options=langues, default=[], key="cat_lang")

            filtres = {
                "title_q": title_q, "use_actor": use_actor, "actor_q": actor_q,
                "use_real": use_real, "real_q": real_q,
                "use_genre": use_genre, "selected_genres": selected_genres,
                "use_year": use_year, "year_min": year_min, "year_max": year_max,
                "use_lang": use_lang, "selected_lang": selected_lang,
            }

        elif choice == "Suggestions":
            st.markdown("""
                <p style='color:#ffffff;font-family:"Cinzel",serif;font-size:1.05rem;
                font-weight:700;letter-spacing:0.05em;
                margin:0 0 10px;text-align:center;'>Réglage des critères</p>
            """, unsafe_allow_html=True)

            # Module compact dans une seule carte (st.container(border=True))
            with st.container(border=True):
                criteria = [
                    ("Box-office", "s_box", "Influence du succès commercial"),
                    ("Genres",     "s_genre", "Priorité aux genres identiques"),
                    ("Casting",    "s_act", "Importance des acteurs/actrices"),
                    ("Réalisation","s_real", "Importance du réalisateur"),
                    ("Note",       "s_note", "Poids lié à la note moyenne")
                ]

                res_poids = {}
                for label, key, help_txt in criteria:
                    curr_val = st.session_state.get(key, "moyen")
                    # Affichage Title ........... Value
                    st.markdown(f"""
                        <div class="slider-header">
                            <span class="slider-label">{label}</span>
                            <span class="slider-value">{curr_val}</span>
                        </div>
                    """, unsafe_allow_html=True)

                    val = st.select_slider(
                        label,
                        options=POIDS_CHOIX,
                        value=curr_val,
                        key=key,
                        help=help_txt,
                        label_visibility="collapsed"
                    )
                    res_poids[key.replace("s_", "")] = val

                poids = res_poids
                st.markdown("""
                    <p style='color:#6870a0;font-size:0.65rem;text-align:center;margin-top:10px;font-style:italic;'>
                        Plus le curseur est fort, plus ce critère influence le modèle.
                    </p>
                """, unsafe_allow_html=True)

    # ── MODAL GLOBALE ─────────────────────────────────────────────────
    if st.session_state.get("modal_active_movie"):
        # On s'assure d'avoir df_prepared, X et model pour le dialogue
        df_prepared = prepare_df_for_ml(df)
        X, model = fit_reco_model(
            df_prepared=df_prepared,
            poids_box=1.0,
            poids_genre=1.0,
            poids_acteurs=1.0,
            poids_real=1.0,
            poids_note=1.0,
        )
        dialog_movie_detail(df, df_prepared, X, model)

    # ── ROUTING ───────────────────────────────────────────────────────
    if choice == "Accueil":
        page_accueil(df)
    elif choice == "Catalogue":
        page_recherche(df, filtres)
    elif choice == "Suggestions":
        page_selection_films_ml(df, poids)
    elif choice == "Statistiques":
        page_kpis(df)
    elif choice == "À propos":
        page_a_propos(df)


if __name__ == "__main__":
    main()
