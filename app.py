import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Plataforma de Inteligencia Educativa - Destino Académico", layout="wide", page_icon="🎓")

def formato_clp(valor):
    if pd.isna(valor): return "$0"
    return f"${valor:,.0f}".replace(",", ".")

# --- 2. CARGA Y PREPARACIÓN DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_csv('SIES_Features_BI_2025_2026_v2.csv')
    
    if 'Score_Atractivo_Integral' not in df.columns:
        min_ing = df['Ingreso_Promedio_4to_Ano'].min()
        max_ing = df['Ingreso_Promedio_4to_Ano'].max()
        ingreso_norm = (df['Ingreso_Promedio_4to_Ano'] - min_ing) / (max_ing - min_ing)
        df['Score_Atractivo_Integral'] = (df['Empleabilidad_1er_Ano'] * 0.4) + (df['Retencion_1er_Ano'] * 0.4) + (ingreso_norm * 100 * 0.2)
        
    df['Pond_PAES'] = 100 - df['Peso_Trayectoria_Escolar_Pct']
    
    if 'Región' in df.columns and 'Region' not in df.columns:
        df['Region'] = df['Región']
        
    return df

df_raw = load_data()

# --- 3. MENÚ LATERAL Y FILTROS GLOBALES ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
st.sidebar.title("Filtros Globales")

if 'Region' in df_raw.columns:
    regiones = ["Todas"] + sorted(df_raw['Region'].dropna().unique().tolist())
    region_sel = st.sidebar.selectbox("📍 Región", regiones)
    df_filtrado = df_raw if region_sel == "Todas" else df_raw[df_raw['Region'] == region_sel]
else:
    sedes = ["Todas"] + sorted(df_raw['Sede'].dropna().unique().tolist())
    region_sel = st.sidebar.selectbox("📍 Sede / Ubicación", sedes)
    df_filtrado = df_raw if region_sel == "Todas" else df_raw[df_raw['Sede'] == region_sel]

areas = ["Todas"] + sorted(df_filtrado['Area_Conocimiento'].dropna().unique().tolist())
area_sel = st.sidebar.selectbox("📚 Área de Conocimiento", areas)
df_filtrado = df_filtrado if area_sel == "Todas" else df_filtrado[df_filtrado['Area_Conocimiento'] == area_sel]

instituciones = ["Todas"] + sorted(df_filtrado['Institucion'].dropna().unique().tolist())
inst_sel = st.sidebar.selectbox("🏛️ Institución", instituciones)
df_final = df_filtrado if inst_sel == "Todas" else df_filtrado[df_filtrado['Institucion'] == inst_sel]

st.sidebar.markdown("---")
st.sidebar.info(f"Mostrando **{len(df_final)}** programas académicos.")

# --- 4. INTERFAZ PRINCIPAL ---
st.title("🎓 Plataforma de Inteligencia Educativa - Destino Académico")
st.markdown("Herramienta analítica integral para estudiantes, apoderados y directivos basada en modelos de Machine Learning.")

tab1, tab2, tab3, tab4 = st.tabs([
    "👨‍🎓 1. Estudiantes (Recomendación)", 
    "👨‍👩‍👧 2. Apoderados (Análisis Financiero)", 
    "🏫 3. Profesores (Evidencia y Admisión)", 
    "📊 4. Jefes UTP (Brechas de Mercado)"
])

# ==========================================
# PESTAÑA 1: ESTUDIANTES
# ==========================================
with tab1:
    st.header("🎯 Orientación Vocacional y Recomendación de Carreras")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    nem_input = col_input1.number_input("Promedio NEM", min_value=4.0, max_value=7.0, value=6.0, step=0.1)
    ranking_input = col_input2.number_input("Puntaje Ranking", min_value=100, max_value=1000, value=700, step=10)
    paes_input = col_input3.number_input("Puntaje PAES Estimado", min_value=100, max_value=1000, value=650, step=10)

    if st.button("Generar Recomendaciones", type="primary"):
        umbral = df_raw['Score_Atractivo_Integral'].quantile(0.70)
        
        # Filtramos y ordenamos TODAS las carreras disponibles
        df_estudiante = df_final[
            (df_final['Score_Atractivo_Integral'].notna()) & 
            (df_final['Ingreso_Promedio_4to_Ano'].notna())
        ].sort_values(by="Score_Atractivo_Integral", ascending=False)
        
        if len(df_estudiante) > 0:
            st.subheader("🏆 Tus Mejores Opciones (Top 3)")
            
            # Extraemos solo las 3 primeras para las tarjetas destacadas
            top_3 = df_estudiante.head(3)
            cols = st.columns(3)
            
            for i, (_, row) in enumerate(top_3.iterrows()):
                es_top = "🌟 ALTA RECOMENDABILIDAD (Top 30%)" if row['Score_Atractivo_Integral'] >= umbral else "Opción Estándar"
                
                with cols[i]:
                    st.info(f"**{row['Nombre_Carrera']}**\n\n_{row['Institucion']}_")
                    st.metric("Score Atractivo", f"{row['Score_Atractivo_Integral']:.1f}/100")
                    st.metric("Empleabilidad", f"{row['Empleabilidad_1er_Ano']:.1f}%")
                    st.metric("Ingreso Proyectado", formato_clp(row['Ingreso_Promedio_4to_Ano']))
                    st.caption(f"Clasificación ML: **{es_top}**")
            
            st.markdown("---")
            
            # Mostramos el resto (si hay más de 3) en una tabla ordenable
            if len(df_estudiante) > 3:
                st.subheader("📋 Otras opciones recomendadas (Ordenadas por Atractivo)")
                
                # Preparamos los datos para que la tabla sea legible
                df_resto = df_estudiante.iloc[3:].copy()
                
                # Formateamos las columnas para la visualización
                df_tabla = df_resto[['Nombre_Carrera', 'Institucion', 'Score_Atractivo_Integral', 'Empleabilidad_1er_Ano', 'Ingreso_Promedio_4to_Ano', 'Costo_Total_Estimado']].copy()
                df_tabla['Score_Atractivo_Integral'] = df_tabla['Score_Atractivo_Integral'].round(1)
                df_tabla['Empleabilidad_1er_Ano'] = df_tabla['Empleabilidad_1er_Ano'].apply(lambda x: f"{x:.1f}%")
                df_tabla['Ingreso_Promedio_4to_Ano'] = df_tabla['Ingreso_Promedio_4to_Ano'].apply(formato_clp)
                df_tabla['Costo_Total_Estimado'] = df_tabla['Costo_Total_Estimado'].apply(formato_clp)
                
                # Renombrar columnas para la interfaz de usuario
                df_tabla.columns = ['Carrera', 'Institución', 'Score Atractivo (0-100)', 'Empleabilidad 1er Año', 'Sueldo 4to Año', 'Costo Total']
                
                st.dataframe(df_tabla, use_container_width=True, hide_index=True)
                
        else:
            st.warning("No hay suficientes datos con los filtros actuales para generar recomendaciones.")

# ==========================================
# PESTAÑA 2: APODERADOS
# ==========================================
with tab2:
    st.header("💰 Análisis Financiero y Retorno de Inversión (ROI)")
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    costo_mediano = df_final['Costo_Total_Estimado'].median()
    payback_mediano = df_final['KPI_Payback_Anios'].median()
    roi_mediano = df_final['KPI_ROI_5A_Pct'].median()
    
    kpi_c1.metric("Costo Total Estimado (Mediana)", formato_clp(costo_mediano))
    color_pb = "normal" if payback_mediano < 3 else "inverse"
    kpi_c2.metric("Payback Period (Años)", f"{payback_mediano:.1f}", delta="Ideal < 3 años", delta_color=color_pb)
    kpi_c3.metric("ROI a 5 Años", f"{roi_mediano:.1f}%")
    
    area_mediana = df_raw[df_raw['Area_Conocimiento'] == area_sel]['Costo_Total_Estimado'].median() if area_sel != "Todas" else df_raw['Costo_Total_Estimado'].median()
    indice_relativo = (costo_mediano / area_mediana) * 100 if pd.notna(area_mediana) and area_mediana > 0 else 100
    kpi_c4.metric("Índice Costo Relativo (vs Área)", f"{indice_relativo:.0f}%", delta="100% = Promedio del mercado", delta_color="off")
    
    st.markdown("---")
    st.subheader("Costo Total vs Ingreso Proyectado (Modelo Regresión Macro)")
    fig_scatter = px.scatter(
        df_final, x="Costo_Total_Estimado", y="Ingreso_Promedio_4to_Ano", 
        color="Tipo_Institucion", hover_name="Nombre_Carrera",
        hover_data={"Institucion": True, "Duracion_Semestres": True},
        labels={"Costo_Total_Estimado": "Costo Total Estimado (CLP)", "Ingreso_Promedio_4to_Ano": "Ingreso 4to Año (CLP)", "Tipo_Institucion": "Tipo"},
        title="Mapa de Inversión Educativa (Burbujas hacia arriba y a la izquierda son mejores)",
        template="plotly_white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# PESTAÑA 3: PROFESORES Y ORIENTADORES
# ==========================================
with tab3:
    st.header("🏫 Evidencia Institucional y Perfiles de Admisión")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Acreditación Promedio", f"{df_final['Acreditacion_Inst_Anos'].mean():.1f} Años")
    col_p2.metric("Tasa de Deserción 1er Año", f"{df_final['KPI_Desercion_1er_Ano_Pct'].mean():.1f}%")
    col_p3.metric("Índice de Sobredemanda", f"{df_final['Indice_Sobredemanda'].mean():.2f}x")
    
    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Segmentación de Estudiantes (ML: PCA + K-Means)")
        df_cluster = df_final[['Promedio_PAES_Matricula', 'Promedio_NEM_Matricula', 'Arancel_Num', 'Costo_Total_Estimado']].dropna()
        if len(df_cluster) > 10:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df_cluster)
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_pca)
            nombres_clusters = {0: "Presupuesto Medio", 1: "Alta Exigencia / Alto Presupuesto", 2: "Orientación Técnica"}
            etiquetas = [nombres_clusters[c] for c in clusters]
            df_plot = pd.DataFrame({'PC1 (Barrera Financiera)': data_pca[:,0], 'PC2 (Exigencia Académica)': data_pca[:,1], 'Arquetipo': etiquetas})
            fig_cluster = px.scatter(df_plot, x='PC1 (Barrera Financiera)', y='PC2 (Exigencia Académica)', color='Arquetipo', title="Arquetipos de Oferta Académica")
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("No hay suficientes datos numéricos completos para generar la segmentación.")
    with col_g2:
        st.subheader("Exigencia: Trayectoria Escolar vs PAES")
        df_pond = df_final.dropna(subset=['Peso_Trayectoria_Escolar_Pct', 'Pond_PAES']).head(15)
        if not df_pond.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(y=df_pond['Nombre_Carrera'], x=df_pond['Peso_Trayectoria_Escolar_Pct'], name='Notas Escolares (NEM+Ranking)', orientation='h', marker_color='#2ca02c'))
            fig_bar.add_trace(go.Bar(y=df_pond['Nombre_Carrera'], x=df_pond['Pond_PAES'], name='Prueba PAES', orientation='h', marker_color='#1f77b4'))
            fig_bar.update_layout(barmode='stack', title="Composición de Ponderaciones (Top 15 Carreras)", yaxis={'autorange': 'reversed'})
            st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PESTAÑA 4: JEFES UTP Y DIRECTIVOS
# ==========================================
with tab4:
    st.header("📊 Brechas de Mercado y Macro-tendencias")
    st.metric("Empleabilidad Promedio al 2do Año", f"{df_final['Empleabilidad_2do_Ano'].mean():.1f}%", delta="Meta >= 85%", delta_color="normal")
    st.markdown("---")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Dispersión de Ingresos y Aranceles")
        fig_box = px.box(df_final, x="Tipo_Institucion", y="Ingreso_Promedio_4to_Ano", color="Tipo_Institucion", title="Varianza Salarial por Tipo de Institución", labels={"Ingreso_Promedio_4to_Ano": "Sueldo 4to Año", "Tipo_Institucion": ""})
        st.plotly_chart(fig_box, use_container_width=True)
    with col_m2:
        st.subheader("Matriz de Correlación del Mercado")
        cols_corr = ['Ingreso_Promedio_4to_Ano', 'Empleabilidad_1er_Ano', 'Duracion_Semestres', 'Arancel_Num']
        df_corr = df_final[cols_corr].dropna().corr()
        fig_heat = px.imshow(df_corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Relación entre variables estratégicas")
        st.plotly_chart(fig_heat, use_container_width=True)
