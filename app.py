import streamlit as st
import pandas as pd
from utils import proceso_total
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="LMB Dashboard", layout="wide")
st.title("📊 Dashboard de la Liga Mexicana de Béisbol")
st.markdown("Este tablero muestra estadísticas generadas automáticamente por el modelo de clustering.")

st.markdown("""
    <style>
    .stButton>button {
        background-color: white;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }

    div[data-testid="stForm"] {
        background-color: #002b54;
        padding: 20px;
        border-radius: 10px;
    }

    h1, h2, h3, h4, h5, h6, h7 {
        color: #9e2100 !important;
    }

    .stTextInput, .stNumberInput, .stDateInput, .stSelectbox, .stCheckbox {
        background-color: #0056b1;
        border: 2px solid #ffffff;
        border-radius: 8px;
        padding: 10px;
        padding-bottom: 20px;
    }

    .stCheckbox {
        background-color: #47a1ff;
        border: 2px solid #ffffff;
        border-radius: 8px;
        padding: 10px;
        padding-bottom: 20px;
    }

    label, .stCheckbox > div {
        color: #9e2100 !important;
        font-weight: bold;
    }

    .stMarkdown p {
        color: #9e2100;
    }

    /* === NUEVO: Fondo blanco para tablas en st.dataframe() === */
    div[data-testid="stDataFrame"] {
        background-color: white !important;
        border-radius: 8px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] iframe {
        background-color: white !important;
    }

    /* === NUEVO: Botón de descarga blanco === */
    div[data-testid="stDownloadButton"] > button {
        background-color: white !important;
        color: black !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: 1px solid #ccc;
    }
    
    table {
        width: 100% !important;
        background-color: white !important;
        color: black !important;
        border-collapse: collapse;
        border-radius: 10px;
    }

    th {
        background-color: black !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px;
        border: 1px solid #ccc;
    }

    td {
        padding: 10px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar session_state si no existe
if "datos_cargados" not in st.session_state:
    st.session_state.datos_cargados = False

# Botón para ejecutar el proceso
if st.button("🔄 Ejecutar proceso y actualizar datos"):
    st.info("Ejecutando proceso completo... puede tardar unos segundos ⏳")
    equipo_local, equipo_visitante, bateo_local, bateo_visita, lanzamiento_local, lanzamiento_visita, Cruces = proceso_total()

    # Guardar en session_state
    st.session_state.equipo_local = equipo_local
    st.session_state.equipo_visitante = equipo_visitante
    st.session_state.bateo_local = bateo_local
    st.session_state.bateo_visita = bateo_visita
    st.session_state.lanzamiento_local = lanzamiento_local
    st.session_state.lanzamiento_visita = lanzamiento_visita
    st.session_state.Cruces = Cruces
    st.session_state.datos_cargados = True

# Si ya se cargaron los datos
if st.session_state.datos_cargados:

    equipos = sorted(st.session_state.equipo_local["Nombre_equipo"].unique())

    # Selección de equipos
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Selecciona equipo local", equipos, key="local_select")
    with col2:
        visita = st.selectbox("Selecciona equipo visitante", equipos, key="visita_select", index=1)

    # Filtrado por equipo
    equipolocal = st.session_state.equipo_local.query("Nombre_equipo == @local")
    bateolocal = st.session_state.bateo_local.query("Nombre_equipo == @local")
    lanzamientolocal = st.session_state.lanzamiento_local.query("Nombre_equipo == @local")

    equipovisitante = st.session_state.equipo_visitante.query("Nombre_equipo == @visita")
    bateovisita = st.session_state.bateo_visita.query("Nombre_equipo == @visita")
    lanzamientovisita = st.session_state.lanzamiento_visita.query("Nombre_equipo == @visita")

    # Selección de jugadores
    with st.container():
        st.markdown("### 🧢 Selección de alineación titular")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### 🏠 {local} - Jugadores titulares")

            bateadores_local = bateolocal["Bateador"].unique()
            lanzadores_local = lanzamientolocal["Lanzador"].unique()

            seleccion_bateadores_local = st.multiselect(
                "Selecciona bateadores locales",
                options=sorted(bateadores_local)
            )

            seleccion_lanzador_local = st.multiselect(
                "Selecciona lanzador local",
                options=sorted(lanzadores_local)
            )

        with col2:
            st.markdown(f"#### 🧳 {visita} - Jugadores titulares")

            bateadores_visita = bateovisita["Bateador"].unique()
            lanzadores_visita = lanzamientovisita["Lanzador"].unique()

            seleccion_bateadores_visita = st.multiselect(
                "Selecciona bateadores visitantes",
                options=sorted(bateadores_visita)
            )

            seleccion_lanzador_visita = st.multiselect(
                "Selecciona lanzador visitante",
                options=sorted(lanzadores_visita)
            )

    # Estadísticas por jugador
    with st.container():
        st.markdown("## 📋 Estadísticas de jugadores seleccionados")

        st.markdown("### 🏠 Equipo Local")

        st.markdown("#### 👥 Bateadores Locales")
        df_bateadores_local = bateolocal[bateolocal["Bateador"].isin(seleccion_bateadores_local)]
        st.table(df_bateadores_local)

        st.markdown("#### 🎯 Lanzador Local")
        df_lanzador_local = lanzamientolocal[lanzamientolocal["Lanzador"].isin(seleccion_lanzador_local)]
        st.table(df_lanzador_local)
        
        # Tabla resumen local
        st.markdown("🔎 **Resumen estadístico - Bateadores Locales**")
        columnas_bateadores = [
            'Carreras', 'Hits', 'Jonrons', 'Carreras.impulsadas', 'Ponches', 'PRO', 'OBP', 'Cluster'
        ]
        resumen_bateo_local = df_bateadores_local[columnas_bateadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_bateo_local.index = [local]
        st.table(resumen_bateo_local)

        st.markdown("🔎 **Resumen estadístico - Lanzador Local**")
        columnas_lanzadores = [
            'EFE', 'BB', 'Jonrons', 'Hits', 'Carreras','PRO','WHIP', 'Cluster'
        ]
        resumen_pitcheo_local = df_lanzador_local[columnas_lanzadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_pitcheo_local.index = [local]
        st.table(resumen_pitcheo_local)

        st.markdown("---")

        st.markdown("### 🧳 Equipo Visitante")

        st.markdown("#### 👥 Bateadores Visitantes")
        df_bateadores_visita = bateovisita[bateovisita["Bateador"].isin(seleccion_bateadores_visita)]
        st.table(df_bateadores_visita)

        st.markdown("#### 🎯 Lanzador Visitante")
        df_lanzador_visita = lanzamientovisita[lanzamientovisita["Lanzador"].isin(seleccion_lanzador_visita)]
        st.table(df_lanzador_visita)
        
        # Tabla resumen visitante
        st.markdown("🔎 **Resumen estadístico - Bateadores Visitantes**")
        resumen_bateo_visita = df_bateadores_visita[columnas_bateadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_bateo_visita.index = [visita]
        st.table(resumen_bateo_visita)
        
        st.markdown("🔎 **Resumen estadístico - Lanzador Visitante**")
        resumen_pitcheo_visita = df_lanzador_visita[columnas_lanzadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_pitcheo_visita.index = [visita]
        st.table(resumen_pitcheo_visita)
        
        st.markdown("---")
        
    st.markdown("## 📊 Comparación gráfica entre equipos")

    # =================== Comparativa BATEADORES ===================
    st.markdown("### 👥 Comparativa de Bateadores")

    # Columnas comunes a comparar
    cols_bateo = ['Hits', 'Jonrons', 'Carreras', 'Carreras.impulsadas', 'Ponches', 'PRO', 'OBP']

    # Filtrar columnas que existan (por si falta alguna)
    cols_bateo = [col for col in cols_bateo if col in resumen_bateo_local.columns]

    fig_bateo = go.Figure()

    fig_bateo.add_trace(go.Bar(
        x=cols_bateo,
        y=[resumen_bateo_local[col].values[0] for col in cols_bateo],
        name=local,
        marker_color='red'
    ))

    fig_bateo.add_trace(go.Bar(
        x=cols_bateo,
        y=[resumen_bateo_visita[col].values[0] for col in cols_bateo],
        name=visita,
        marker_color='blue'
    ))

    fig_bateo.update_layout(
    barmode='group',
    title="Comparativa Bateadores (promedios)",
    yaxis_title="Valor Promedio",
    xaxis_title="Estadísticas",
    template="plotly_white",  # Aún sirve para mantener el estilo base

    # 👇 Estilo del fondo y texto
    paper_bgcolor="#17106b",   # Fondo del área fuera del gráfico
    plot_bgcolor="#002a51",    # Fondo dentro del gráfico

    font=dict(
        color='white',         # Color del texto
        family='Arial',        # Tipo de letra
        size=14
    ),

    title_font=dict(
        color='white',
        size=20,
        family='Arial',
        # No hay weight='bold', pero Plotly usa negrita por defecto en títulos
    ),

    xaxis=dict(
        title_font=dict(color='white', size=16),
        tickfont=dict(color='white', size=14),
    ),

    yaxis=dict(
        title_font=dict(color='white', size=16),
        tickfont=dict(color='white', size=14),
    ),

    legend=dict(
        font=dict(color='white', size=14)
    )
    )

    st.plotly_chart(fig_bateo, use_container_width=True)

    # =================== Comparativa LANZADORES ===================
    st.markdown("### 🎯 Comparativa de Lanzadores")

    cols_pitcheo = ['EFE', 'BB', 'Jonrons', 'Hits', 'Carreras','PRO','WHIP']

    cols_pitcheo = [col for col in cols_pitcheo if col in resumen_pitcheo_local.columns]

    fig_pitcheo = go.Figure()

    fig_pitcheo.add_trace(go.Bar(
        x=cols_pitcheo,
        y=[resumen_pitcheo_local[col].values[0] for col in cols_pitcheo],
        name=local,
        marker_color='red'
    ))

    fig_pitcheo.add_trace(go.Bar(
        x=cols_pitcheo,
        y=[resumen_pitcheo_visita[col].values[0] for col in cols_pitcheo],
        name=visita,
        marker_color='blue'
    ))
    
    fig_pitcheo.update_layout(
    barmode='group',
    title="Comparativa Lanzadores (promedios)",
    yaxis_title="Valor Promedio",
    xaxis_title="Estadísticas",
    template="plotly_white",  # Aún sirve para mantener el estilo base

    # 👇 Estilo del fondo y texto
    paper_bgcolor="#17106b",   # Fondo del área fuera del gráfico
    plot_bgcolor="#002a51",    # Fondo dentro del gráfico

    font=dict(
        color='white',         # Color del texto
        family='Arial',        # Tipo de letra
        size=14
    ),

    title_font=dict(
        color='white',
        size=20,
        family='Arial',
        # No hay weight='bold', pero Plotly usa negrita por defecto en títulos
    ),

    xaxis=dict(
        title_font=dict(color='white', size=16),
        tickfont=dict(color='white', size=14),
    ),

    yaxis=dict(
        title_font=dict(color='white', size=16),
        tickfont=dict(color='white', size=14),
    ),

    legend=dict(
        font=dict(color='white', size=14)
    )
    )

    st.plotly_chart(fig_pitcheo, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## 🔮 Probabilidad calculada basada en alineaciones seleccionadas")

    # Filtramos la base de cruces según selección

    # Local batea vs. visitante lanza
    
    # Validación por si no se selecciona nada
    if not seleccion_bateadores_local:
        seleccion_bateadores_local = list(bateolocal["Bateador"].unique())

    if not seleccion_lanzador_local:
        seleccion_lanzador_local = list(lanzamientolocal["Lanzador"].unique())

    if not seleccion_bateadores_visita:
        seleccion_bateadores_visita = list(bateovisita["Bateador"].unique())

    if not seleccion_lanzador_visita:
        seleccion_lanzador_visita = list(lanzamientovisita["Lanzador"].unique())
    
    df_cruces = st.session_state.Cruces
    
    cruce1 = df_cruces[
        df_cruces["JUGADORJUGADOR_x"].isin(seleccion_bateadores_local) &
        df_cruces["JUGADORJUGADOR_y"].isin(seleccion_lanzador_visita)
    ]
    PROBA1 = cruce1["Proba_ajustada"].mean()

    # Visitante batea vs. local lanza
    cruce2 = df_cruces[
        df_cruces["JUGADORJUGADOR_x"].isin(seleccion_bateadores_visita) &
        df_cruces["JUGADORJUGADOR_y"].isin(seleccion_lanzador_local)
    ]
    PROBA2 = cruce2["Proba_ajustada"].mean()

    # Calculamos PWL y PWV
    PWL = 1 - (PROBA1 / (PROBA1 + PROBA2))
    PWV = 1 - PWL

    # Mostramos

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Partido"],
        x=[PWL],
        name=local,
        orientation='h',
        marker=dict(color='crimson'),
        text=f"{PWL:.1%}",
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(color='white', size=20)
    ))

    # Barra del equipo visitante
    fig.add_trace(go.Bar(
        y=["Partido"],
        x=[PWV],
        name=visita,
        orientation='h',
        marker=dict(color='royalblue'),
        text=f"{PWV:.1%}",
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(color='white', size=30)
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(showticklabels=False),
        template="plotly_white",
        height=200,  # ← más alto
        margin=dict(l=50, r=50, t=60, b=30),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
        

    # Descarga
    st.markdown("---")
    
    st.download_button(
        label="⬇ Descargar estadisticas bateadores locales",
        data = df_bateadores_local.to_csv(index=False),
        file_name="resumen_bateo_local_download.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="⬇ Descargar estadisticas bateadores visitantes",
        data = df_bateadores_visita.to_csv(index=False),
        file_name="resumen_bateo_visita_download.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="⬇ Descargar estadisticas lanzadores locales",
        data = df_lanzador_local.to_csv(index=False),
        file_name="resumen_pitcheo_local_download.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="⬇ Descargar estadisticas lanzadores visitantes",
        data = df_lanzador_visita.to_csv(index=False),
        file_name="resumen_pitcheo_visita_download.csv",
        mime="text/csv"
    )

else:
    st.warning("Haz clic en el botón para cargar los datos.")
    
