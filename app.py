import streamlit as st
import pandas as pd
from utils import proceso_total
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="LMB Dashboard", layout="wide")

# Encabezado
col1, col2 = st.columns([8, 3])
with col1:
    st.title("📊 Tablero de la Liga Mexicana de Béisbol")
with col2:
    st.image("qr.jpg", width=200)

st.markdown("Este tablero muestra estadísticas generadas automáticamente por el modelo de clustering.")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #878B36; /* Verde maguey */
        color: #3E2C20;            /* Marrón oscuro */
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: 2px solid #3E2C20;
    }

    /* Fondo del formulario */
    div[data-testid="stForm"] {
        background-color: #F4E1A0; /* Beige claro */
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #6C593C;
    }

    h1, h2, h3, h4, h5, h6, h7 {
        color: #3E2C20 !important; /* Marrón oscuro */
    }

    /* Inputs del formulario */
    .stTextInput, .stNumberInput, .stDateInput, .stSelectbox {
        background-color: #FFF6D6; /* Tonalidad clara artesanal */
        border: 2px solid #A3472F; /* Terracota para destacar */
        border-radius: 8px;
        padding: 10px;
        color: #3E2C20;
    }

    .stCheckbox {
        background-color: #A3472F; /* Fondo terracota */
        border: 2px solid #3E2C20;
        border-radius: 8px;
        padding: 10px;
        color: #F4E1A0;
    }

    label, .stCheckbox > div {
        color: #3E2C20 !important;
        font-weight: bold;
    }

    .stMarkdown p {
        color: #3E2C20;
    }

    /* Fondo general */
    .main {
        background-color: #F4E1A0;
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
        st.dataframe(df_bateadores_local, use_container_width=True)

        st.markdown("#### 🎯 Lanzador Local")
        df_lanzador_local = lanzamientolocal[lanzamientolocal["Lanzador"].isin(seleccion_lanzador_local)]
        st.dataframe(df_lanzador_local, use_container_width=True)
        
        # Tabla resumen local
        st.markdown("🔎 **Resumen estadístico - Bateadores Locales**")
        columnas_bateadores = [
            'Carreras', 'Hits', 'Jonrons', 'Carreras.impulsadas', 'Ponches', 'PRO', 'OBP', 'Cluster'
        ]
        resumen_bateo_local = df_bateadores_local[columnas_bateadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_bateo_local.index = [local]
        st.dataframe(resumen_bateo_local, use_container_width=True)

        st.markdown("🔎 **Resumen estadístico - Lanzador Local**")
        columnas_lanzadores = [
            'EFE', 'BB', 'Jonrons', 'Hits', 'Carreras','PRO','WHIP', 'Cluster'
        ]
        resumen_pitcheo_local = df_lanzador_local[columnas_lanzadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_pitcheo_local.index = [local]
        st.dataframe(resumen_pitcheo_local, use_container_width=True)

        st.markdown("---")

        st.markdown("### 🧳 Equipo Visitante")

        st.markdown("#### 👥 Bateadores Visitantes")
        df_bateadores_visita = bateovisita[bateovisita["Bateador"].isin(seleccion_bateadores_visita)]
        st.dataframe(df_bateadores_visita, use_container_width=True)

        st.markdown("#### 🎯 Lanzador Visitante")
        df_lanzador_visita = lanzamientovisita[lanzamientovisita["Lanzador"].isin(seleccion_lanzador_visita)]
        st.dataframe(df_lanzador_visita, use_container_width=True)
        
        # Tabla resumen visitante
        st.markdown("🔎 **Resumen estadístico - Bateadores Visitantes**")
        resumen_bateo_visita = df_bateadores_visita[columnas_bateadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_bateo_visita.index = [visita]
        st.dataframe(resumen_bateo_visita, use_container_width=True)
        
        st.markdown("🔎 **Resumen estadístico - Lanzador Visitante**")
        resumen_pitcheo_visita = df_lanzador_visita[columnas_lanzadores].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
        resumen_pitcheo_visita.index = [visita]
        st.dataframe(resumen_pitcheo_visita, use_container_width=True)
        
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
        marker_color="#D8B000"
    ))

    fig_bateo.add_trace(go.Bar(
        x=cols_bateo,
        y=[resumen_bateo_visita[col].values[0] for col in cols_bateo],
        name=visita,
        marker_color='#878B36'
    ))

    fig_bateo.update_layout(
        barmode='group',
        title="Comparativa Bateadores (promedios)",
        yaxis_title="Valor Promedio",
        xaxis_title="Estadísticas",
        template="plotly_white",
        legend=dict(
            font=dict(
                size=16  # Puedes ajustar este valor a lo que necesites
            )
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
        marker_color="#D8B000"
    ))
    fig_pitcheo.add_trace(go.Bar(
        x=cols_pitcheo,
        y=[resumen_pitcheo_visita[col].values[0] for col in cols_pitcheo],
        name=visita,
        marker_color='#878B36'
    ))

    fig_pitcheo.update_layout(
        barmode='group',
        title="Comparativa Lanzadores (promedios)",
        yaxis_title="Valor Promedio",
        xaxis_title="Estadísticas",
        template="plotly_white",
        legend=dict(
            font=dict(
                size=16  # Puedes ajustar este valor a lo que necesites
            )
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
        marker=dict(color='#D8B000'),
        text=f"{PWL:.1%}",
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(color='white', size=30)
    ))

    # Barra del equipo visitante
    fig.add_trace(go.Bar(
        y=["Partido"],
        x=[PWV],
        name=visita,
        orientation='h',
        marker=dict(color='#878B36'),
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
    st.warning("Haz clic en el botón para comenzar.")