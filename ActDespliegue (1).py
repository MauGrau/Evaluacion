import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import re
import plotly.graph_objects as go


#CONFIG

st.set_page_config(page_title="Airbnb — Plantilla Adaptada", layout="wide")


st.sidebar.title("AIRBNB")
st.sidebar.subheader("Ciudades")

DEFAULTS = [
    ("listingsBarcelona.csv", "Barcelona"),
    ("listingsAmsterdam.csv", "Amsterdam"),
    ("listingsMilan.csv", "Milan"),
    ("listingsParis.csv", "Paris"),
    ("listingsMadrid.csv", "Madrid"),
]

paths, names = [], []
for i, (ruta_def, nombre_def) in enumerate(DEFAULTS, start=1):
    ruta = st.sidebar.text_input(f"Ruta ciudad {i}", value=ruta_def, key=f"ruta{i}")
    nombre = st.sidebar.text_input(f"Nombre ciudad {i}", value=nombre_def, key=f"nombre{i}")
    if i < len(DEFAULTS):
        st.sidebar.divider()
    if ruta.strip():
        paths.append(ruta.strip())
        names.append((nombre or nombre_def).strip())

if not paths:
    st.info("Define al menos una ruta válida en la barra lateral.")
    st.stop()


#UTILIDADES DE CARGA / LIMPIEZA
def _to_float_price(s):
    if pd.isna(s): return np.nan
    s = str(s)
    s = re.sub(r"[^\d\.\-]", "", s) 
    try:
        return float(s) if s != "" else np.nan
    except:
        return np.nan

def _bathrooms_from_text(txt):
    if pd.isna(txt): return np.nan
    s = str(txt).lower()
    if "half" in s and not re.search(r"\d+(\.\d+)?", s):
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def limpiar_estandarizar(df: pd.DataFrame, ciudad: str) -> pd.DataFrame:
    d = df.copy()
    d["ciudad"] = ciudad
    if "id" not in d.columns:
        d["id"] = np.arange(len(d)) + 1

    d["price"] = d.get("price", np.nan)
    d["price"] = d["price"].map(_to_float_price)

    if "neighbourhood_cleansed" in d.columns:
        d["barrio_std"] = d["neighbourhood_cleansed"]
    elif "neighbourhood" in d.columns:
        d["barrio_std"] = d["neighbourhood"]
    else:
        d["barrio_std"] = np.nan

    d["room_type"] = d.get("room_type", pd.Series(index=d.index, dtype="object")).astype(str).str.strip().replace({"nan": np.nan})
    d["accommodates"] = pd.to_numeric(d.get("accommodates", np.nan), errors="coerce")

    if "bathrooms_text" in d.columns:
        d["bathrooms_num"] = d["bathrooms_text"].map(_bathrooms_from_text)
    else:
        d["bathrooms_num"] = pd.to_numeric(d.get("bathrooms", np.nan), errors="coerce")

    d["latitude"]  = pd.to_numeric(d.get("latitude", np.nan), errors="coerce")
    d["longitude"] = pd.to_numeric(d.get("longitude", np.nan), errors="coerce")

    if "amenities" in d.columns:
        d["amenities_count"] = d["amenities"].astype(str).apply(
            lambda x: 0 if x in ("nan", "", "[]") else len([a for a in re.split(r"[,\|]", x.strip("[]")) if a.strip()]))
    else:
        d["amenities_count"] = np.nan

    d["price_per_person"] = np.where((d["accommodates"] >= 1) & d["price"].notna(),
                                     d["price"] / d["accommodates"], np.nan)

    cols = ["id","ciudad","barrio_std","room_type","accommodates","bathrooms_num",
            "price","price_per_person","amenities_count","latitude","longitude",
            "property_type","host_is_superhost","cancellation_policy",
            "instant_bookable","review_scores_rating","number_of_reviews",
            "bed_type","neighbourhood_group_cleansed",
            "require_guest_profile_picture","require_guest_phone_verification",
            "host_response_time","host_identity_verified","has_availability","source"]
    keep = [c for c in cols if c in d.columns]
    return d[keep]

def recortar_outliers_por_ciudad(df: pd.DataFrame, col="price", p_low=0.01, p_high=0.99):
    limpio = []
    for ciudad, g in df.groupby("ciudad", dropna=False):
        if g[col].notna().sum() < 50:
            limpio.append(g); continue
        low, high = g[col].quantile(p_low), g[col].quantile(p_high)
        limpio.append(g[(g[col].isna()) | ((g[col] >= low) & (g[col] <= high))])
    return pd.concat(limpio, ignore_index=True)

def safe_cut(s, bins, labels):
    try:
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    except Exception:
        return pd.Series(index=s.index, dtype="category")


# PLANTILLA: load_data() (con @st.cache_resource)

@st.cache_data(show_spinner=False)
def load_data(paths: list[str], names: list[str]):
    HERE = Path(__file__).resolve().parent

    # 1) Leer CSVs
    partes = []
    for path, city in zip(paths, names):
        p = Path(path)
        if not p.is_absolute():
            p = HERE / p
        df_raw = pd.read_csv(p, low_memory=False)
        partes.append(limpiar_estandarizar(df_raw, city))

    # 2) Unificar + deduplicar
    df_all = pd.concat(partes, ignore_index=True)
    if {"ciudad","id"}.issubset(df_all.columns):
        df_all = (df_all
                  .sort_values(["ciudad","id"])
                  .drop_duplicates(subset=["ciudad","id"], keep="first"))

    # 3) Recortar outliers por ciudad (precio)
    if "price" in df_all.columns:
        df_all = recortar_outliers_por_ciudad(df_all, col="price", p_low=0.01, p_high=0.99)

    # 4) (AÚN SIN features de bandas aquí; solo limpieza base)
    return df_all

df = load_data(paths, names)
candidatas = [
    "room_type","barrio_std","property_type","instant_bookable","cancellation_policy",
    "host_is_superhost","host_identity_verified","host_response_time","has_availability",
    "bed_type","source","neighbourhood_group_cleansed",
    "price_range","accommodates_band","bathrooms_band","amenities_band",
]

Lista = [c for c in candidatas if c in df.columns]

# Si faltan para llegar a 15, completa con categóricas auto-detectadas (baja cardinalidad)
if len(Lista) < 15:
    auto = [
        c for c in df.select_dtypes(include=["object","category"]).columns
        if c not in Lista and df[c].nunique(dropna=False) <= 50
    ]
    Lista = (Lista + auto)[:15]

# Validación
if not Lista:
    st.error("No encontré variables categóricas. Revisa que el DataFrame tenga columnas tipo object/category.")
    st.stop()



#MENÚ 

st.sidebar.title("Tipo de analisis")
View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=["Extracción de Características", "Regresión Lineal", "Regresión No Lineal", "Regresión Logística"]
)


#VISTA 1:EXTRACCIÓN DE CARACTERÍSTICAS

if View == "Extracción de Características":
    st.title("Extracción de Características — Airbnb")

    # Selectores extra 
    ciudad_sel = st.sidebar.selectbox("Ciudad asignada", sorted(df["ciudad"].dropna().unique().tolist()))
    top_k = st.sidebar.slider("Top categorías por gráfica", 5, 30, 10)
    mostrar_tabla = st.sidebar.checkbox("Mostrar tabla de frecuencias", value=False)

    #Variable categórica 
    Variable_Cat = st.sidebar.selectbox("Variables", options=Lista)

    #Subconjunto por ciudad
    df_city = df[df["ciudad"] == ciudad_sel].copy()

    #Tabla de frecuencias
    Tabla_frecuencias = (
        df_city[Variable_Cat]
        .astype("object").fillna("NA").astype(str)
        .value_counts().head(top_k)
        .reset_index()
    )
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    #layout
    cA, cB = st.columns(2)
    with cA:
        st.write("Gráfico de Barras")
        fig1 = px.bar(Tabla_frecuencias, x='categorias', y='frecuencia',
                      title=f'Frecuencia por categoría — {Variable_Cat} ({ciudad_sel})')
        fig1.update_xaxes(automargin=True)
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with cB:
        st.write("Gráfico de Pastel")
        fig2 = px.pie(Tabla_frecuencias, names='categorias', values='frecuencia',
                      title=f'Frecuencia por categoría — {Variable_Cat} ({ciudad_sel})')
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        st.write("Gráfico de anillo (dona)")
        fig3 = px.pie(Tabla_frecuencias, names='categorias', values='frecuencia',
                      hole=0.45, title=f'Dona — {Variable_Cat} ({ciudad_sel})')
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with cD:
        st.write("Gráfico de área")
        tmp_area = Tabla_frecuencias.sort_values("categorias")
        fig4 = px.area(tmp_area, x='categorias', y='frecuencia',
                       title=f'Área — {Variable_Cat} ({ciudad_sel})')
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)

    
    st.markdown("---")
    cE, cF = st.columns(2)

    # BOXPLOT
    with cE:
        if "price" in df_city.columns and df_city["price"].notna().any():
            # Elige contra qué categorizar el precio
            cat_para_box = st.selectbox("Categoría para Boxplot (precio)", options=[c for c in Lista if c in df_city.columns], index=0, key="boxcat")
            df_box = df_city[[cat_para_box, "price"]].dropna()
            # limitar categorías al Top K más frecuentes para mejor lectura
            top_cats = df_box[cat_para_box].astype("object").value_counts().head(min(top_k, 15)).index
            df_box = df_box[df_box[cat_para_box].astype("object").isin(top_cats)]
            fig_box = px.box(df_box, x=cat_para_box, y="price", points=False,
                             title=f"Boxplot de precio por {cat_para_box} — {ciudad_sel}")
            fig_box.update_layout(height=480)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No hay columna de precio válida para el boxplot.")

    # HEATMAP: coocurrencias entre 2 categóricas
    with cF:
        # escoger dos categóricas (de la lista) para mapa de calor de frecuencias
        cats_heat = [c for c in Lista if c in df_city.columns]
        if len(cats_heat) >= 2:
            cat_x = st.selectbox("Heatmap — Eje X", options=cats_heat, index=0, key="hx")
            cat_y = st.selectbox("Heatmap — Eje Y", options=cats_heat, index=min(1, len(cats_heat)-1), key="hy")
            t = (
                df_city[[cat_x, cat_y]].astype("object").fillna("NA")
                .value_counts().reset_index(name="freq")
            )
            # Quedarnos con Top K por eje para legibilidad
            top_x = t[cat_x].value_counts().head(min(top_k, 15)).index
            top_y = t[cat_y].value_counts().head(min(top_k, 15)).index
            t = t[t[cat_x].isin(top_x) & t[cat_y].isin(top_y)]

            fig_hm = px.density_heatmap(
                t, x=cat_x, y=cat_y, z="freq", color_continuous_scale="Blues",
                title=f"Heatmap de frecuencias — {cat_x} vs {cat_y} ({ciudad_sel})"
            )
            fig_hm.update_layout(height=480)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Selecciona al menos dos variables categóricas para el heatmap.")

    # TABLA de frecuencias (toggle)
    if mostrar_tabla:
        st.markdown("### Tabla de frecuencias")
        # Asegura índice desde 1
        Tabla_frecuencias = Tabla_frecuencias.reset_index(drop=True)
        Tabla_frecuencias.index = np.arange(1, len(Tabla_frecuencias) + 1)
        Tabla_frecuencias.index.name = "#"
        st.dataframe(Tabla_frecuencias, use_container_width=True)

    # === ANÁLISIS GEOESPACIAL ===
    if {"latitude", "longitude"}.issubset(df_city.columns):
        st.markdown("---")
        st.subheader("Análisis Geoespacial")
        
        df_geo = df_city[["latitude", "longitude", "price", "barrio_std"]].dropna()
        
        if len(df_geo) > 0:
            col_geo1, col_geo2 = st.columns([3, 1])
            
            with col_geo2:
                color_by_geo = st.selectbox("Colorear por:", ["price", "barrio_std"], key="color_geo")
                map_style = st.selectbox("Estilo de mapa:", 
                                       ["open-street-map", "carto-positron", "carto-darkmatter"], 
                                       key="map_style")
            
            with col_geo1:
                if color_by_geo == "price":
                    fig_map = px.scatter_mapbox(
                        df_geo, lat="latitude", lon="longitude", 
                        color="price", size="price",
                        hover_data=["barrio_std"],
                        mapbox_style=map_style,
                        title=f"Distribución Geográfica por Precio - {ciudad_sel}",
                        height=500,
                        color_continuous_scale="Viridis"
                    )
                else:
                    fig_map = px.scatter_mapbox(
                        df_geo, lat="latitude", lon="longitude", 
                        color="barrio_std",
                        hover_data=["price"],
                        mapbox_style=map_style,
                        title=f"Distribución Geográfica por Barrio - {ciudad_sel}",
                        height=500
                    )
                
                # Centrar el mapa
                center_lat = df_geo["latitude"].median()
                center_lon = df_geo["longitude"].median()
                fig_map.update_layout(
                    mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10)
                )
                st.plotly_chart(fig_map, use_container_width=True)


