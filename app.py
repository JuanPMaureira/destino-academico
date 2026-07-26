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

import joblib

# --- CARGA DE MODELOS EN CACHÉ ---
@st.cache_resource
def load_models():
    # Asegúrate de que los archivos .pkl estén en la misma carpeta o subidos a Colab
    return (
        joblib.load('modelo_clasificacion.pkl'),
        joblib.load('modelo_regresion.pkl'),
        joblib.load('modelo_clustering.pkl')
    )

modelo_clasificacion, modelo_regresion, modelo_clustering = load_models()

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
# PESTAÑA 1: ESTUDIANTES (VERSIÓN MACHINE LEARNING)
# ==========================================
with tab1:
    st.header("🎯 Orientación Vocacional y Proyección de Ingresos (Modelos Predictivos)")
    
    with st.expander("ℹ️ ¿Cómo funciona este sistema de recomendaciones?"):
        st.markdown("""
        * **Score Atractivo (Base 0-100):** Índice ponderado de Empleabilidad (40%), Retención (40%) y Sueldo (20%).
        * **Sueldo Proyectado (ML):** Proyección calculada por Inteligencia Artificial (Random Forest) para tu 4to año de titulación.
        * **Clasificación ML:** El algoritmo determina si una carrera pertenece al **Top 30% más rentable y seguro** (🌟 Alta Recomendabilidad).
        """)
    
    admision_directa = st.checkbox("☑️ Admisión Directa IP/CFT (No rendiré PAES / No aplica para mi carrera)")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    nem_input = col_input1.number_input("Promedio NEM", min_value=4.0, max_value=7.0, value=6.0, step=0.1, help="Promedio de enseñanza media.")
    ranking_input = col_input2.number_input("Puntaje Ranking", min_value=100, max_value=1000, value=700, step=10)
    
    if admision_directa:
        paes_input = 0
        col_input3.info("PAES omitido (Buscando solo IP, CFT y carreras sin PAES)")
    else:
        paes_input = col_input3.number_input("Puntaje PAES Estimado", min_value=100, max_value=1000, value=650, step=10)

    if st.button("Generar Recomendaciones Predictivas", type="primary"):
        if len(df_final) > 0:
            df_pred = df_final.copy()
            
            # --- FILTRO 1: TIPO DE ADMISIÓN (IP/CFT/Sin PAES) ---
            if admision_directa:
                # Si marcó admisión directa, dejamos solo instituciones que no sean Universidades 
                # o carreras que reporten explícitamente 0% de exigencia PAES.
                condicion_admision = (df_pred['Tipo_Institucion'].isin(['Instituto Profesional', 'Centro de Formación Técnica'])) | (df_pred['Pond_PAES'] == 0)
                df_pred = df_pred[condicion_admision]
            
            # --- FILTRO 2: VIABILIDAD ACADÉMICA (LÓGICA DE NEGOCIO) ---
            if len(df_pred) > 0:
                condicion_nem = (df_pred['Promedio_NEM_Matricula'].isna()) | (df_pred['Promedio_NEM_Matricula'] <= nem_input + 0.3)
                df_pred = df_pred[condicion_nem]
                
                if not admision_directa:
                    condicion_paes = (df_pred['Promedio_PAES_Matricula'].isna()) | (df_pred['Promedio_PAES_Matricula'] <= paes_input + 50)
                    df_pred = df_pred[condicion_paes]
                
            if len(df_pred) == 0:
                st.error("📉 Con los puntajes ingresados o filtros aplicados, no encontramos carreras compatibles. Intenta modificar tus filtros globales.")
            else:
                # --- CONSTRUCCIÓN DE VECTORES DE INFERENCIA ---
                X_clas = pd.DataFrame({
                    'Promedio_PAES_Matricula': float(paes_input),
                    'Arancel_Num': df_pred['Arancel_Num'].astype(float),
                    'Duracion_Semestres': df_pred['Duracion_Semestres'].astype(float),
                    'Promedio_NEM_Matricula': float(nem_input)
                })
                
                X_reg = pd.DataFrame({
                    'Arancel_Num': df_pred['Arancel_Num'].astype(float),
                    'Duracion_Semestres': df_pred['Duracion_Semestres'].astype(float),
                    'Promedio_NEM_Matricula': float(nem_input),
                    'Tipo_Institucion': df_pred['Tipo_Institucion']
                })
                
                df_pred['Recomendado_ML'] = modelo_clasificacion.predict(X_clas)
                df_pred['Ingreso_Proyectado_ML'] = modelo_regresion.predict(X_reg)
                
                df_estudiante = df_pred.sort_values(
                    by=['Recomendado_ML', 'Score_Atractivo_Integral'], 
                    ascending=[False, False]
                )
                
                st.subheader("🏆 Tus Mejores Opciones Viables (Top 3)")
                top_3 = df_estudiante.head(3)
                cols = st.columns(3)
                
                for i, (_, row) in enumerate(top_3.iterrows()):
                    es_top = "🌟 ALTA RECOMENDABILIDAD" if row['Recomendado_ML'] == 1 else "Estándar"
                    
                    with cols[i]:
                        st.info(f"**{row['Nombre_Carrera']}**\n\n_{row['Institucion']}_")
                        st.metric("Score Atractivo (Base)", f"{row['Score_Atractivo_Integral']:.1f}/100", help="Basado en Empleabilidad, Retención y Sueldos.")
                        st.metric("Sueldo 4to Año (ML)", formato_clp(row['Ingreso_Proyectado_ML']), help="Proyectado por nuestro modelo Random Forest.")
                        st.caption(f"Clasificación Algorítmica: **{es_top}**")
                
                st.markdown("---")
                if len(df_estudiante) > 3:
                    st.subheader("📋 Otras opciones evaluadas acordes a tu perfil")
                    df_resto = df_estudiante.iloc[3:].copy()
                    
                    df_tabla = df_resto[['Nombre_Carrera', 'Institucion', 'Recomendado_ML', 'Score_Atractivo_Integral', 'Ingreso_Proyectado_ML', 'Costo_Total_Estimado']].copy()
                    
                    df_tabla['Recomendado_ML'] = df_tabla['Recomendado_ML'].apply(lambda x: "Sí 🌟" if x == 1 else "No")
                    df_tabla['Score_Atractivo_Integral'] = df_tabla['Score_Atractivo_Integral'].round(1)
                    df_tabla['Ingreso_Proyectado_ML'] = df_tabla['Ingreso_Proyectado_ML'].apply(formato_clp)
                    df_tabla['Costo_Total_Estimado'] = df_tabla['Costo_Total_Estimado'].apply(formato_clp)
                    
                    df_tabla.columns = ['Carrera', 'Institución', 'Recomendado por ML', 'Score Base (0-100)', 'Sueldo 4to Año (ML)', 'Costo Total']
                    st.dataframe(df_tabla, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay suficientes datos con los filtros globales seleccionados.")
# ==========================================
# PESTAÑA 2: APODERADOS
# ==========================================
with tab2:
    st.header("💰 Análisis Financiero y Retorno de Inversión (ROI)")
    
    if len(df_final) > 0:
        # 1. Selector específico de Carrera
        st.subheader("Análisis Específico por Programa Académico")
        
        # Creamos una etiqueta combinada para evitar confusiones si hay carreras con el mismo nombre en distintas sedes
        df_final['Carrera_Etiqueta'] = df_final['Nombre_Carrera'] + " - " + df_final['Institucion']
        carreras_disponibles = sorted(df_final['Carrera_Etiqueta'].dropna().unique())
        
        carrera_sel = st.selectbox("Selecciona un programa para analizar su rentabilidad:", carreras_disponibles)
        
        # 2. Filtrar datos de la carrera seleccionada
        df_carrera = df_final[df_final['Carrera_Etiqueta'] == carrera_sel]
        
        # Obtener el área de conocimiento de esta carrera para comparar contra el benchmark (promedio de la industria)
        area_actual = df_carrera['Area_Conocimiento'].iloc[0]
        df_area_benchmark = df_raw[df_raw['Area_Conocimiento'] == area_actual]
        
        # 3. Cálculos de KPIs y Deltas
        costo_carrera = df_carrera['Costo_Total_Estimado'].median()
        costo_area = df_area_benchmark['Costo_Total_Estimado'].median()
        delta_costo = costo_carrera - costo_area
        
        payback_carrera = df_carrera['KPI_Payback_Anios'].median()
        payback_area = df_area_benchmark['KPI_Payback_Anios'].median()
        delta_payback = payback_carrera - payback_area
        
        roi_carrera = df_carrera['KPI_ROI_5A_Pct'].median()
        roi_area = df_area_benchmark['KPI_ROI_5A_Pct'].median()
        delta_roi = roi_carrera - roi_area
        
        indice_carrera = (costo_carrera / costo_area) * 100 if costo_area > 0 else 100
        delta_indice = indice_carrera - 100
        
        # 4. Renderizado de Tarjetas de KPI con benchmark
        st.caption(f"Comparando contra la mediana de su Área de Conocimiento: **{area_actual}**")
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        
        # delta_color="inverse" pone en verde cuando el valor baja (ideal para costos y payback)
        kpi_c1.metric(
            "Costo Total Estimado", 
            formato_clp(costo_carrera), 
            delta=formato_clp(delta_costo), 
            delta_color="inverse", 
            help="Costo de aranceles + titulación. El indicador inferior muestra la diferencia con el promedio del área."
        )
        
        kpi_c2.metric(
            "Payback Period (Años)", 
            f"{payback_carrera:.1f}", 
            delta=f"{delta_payback:.1f} años", 
            delta_color="inverse"
        )
        
        kpi_c3.metric(
            "ROI a 5 Años", 
            f"{roi_carrera:.1f}%", 
            delta=f"{delta_roi:.1f}%", 
            delta_color="normal" # normal: mayor es verde
        )
        
        kpi_c4.metric(
            "Índice Costo Relativo", 
            f"{indice_carrera:.0f}%", 
            delta=f"{delta_indice:.0f}% vs Promedio", 
            delta_color="inverse"
        )
        
        st.markdown("---")
        st.subheader("Contexto: Costo Total vs Ingreso Proyectado (Mercado Seleccionado)")
        
        # Gráfico de dispersión utilizando el df_final completo para ver dónde se ubica la carrera seleccionada
        fig_scatter = px.scatter(
            df_final, x="Costo_Total_Estimado", y="Ingreso_Promedio_4to_Ano", 
            color="Tipo_Institucion", hover_name="Nombre_Carrera",
            hover_data={"Institucion": True, "Duracion_Semestres": True},
            labels={"Costo_Total_Estimado": "Costo Total (CLP)", "Ingreso_Promedio_4to_Ano": "Ingreso 4to Año (CLP)"},
            title="Mapa de Inversión Educativa (Burbujas hacia arriba y a la izquierda son mejores)",
            template="plotly_white"
        )
        # Añadir marcador para destacar la carrera seleccionada en el scatter plot
        fig_scatter.add_vline(x=costo_carrera, line_dash="dot", line_color="gray", annotation_text="Costo Carrera Seleccionada")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No hay datos disponibles para mostrar el análisis financiero.")

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
            fig_cluster = px.scatter(df_plot, x='PC1 (Barrera Financiera)', y='PC2 (Exigencia Académica)', color='Arquetipo')
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("No hay suficientes datos numéricos para segmentación algorítmica.")

    with col_g2:
        st.subheader("Exigencia Académica Dinámica: NEM+Ranking vs PAES")
        # Filtro y ordenamiento dinámico del TOP 15 basado en el df_final filtrado de la barra lateral
        df_pond = df_final.dropna(subset=['Peso_Trayectoria_Escolar_Pct', 'Pond_PAES'])
        df_pond = df_pond.sort_values(by='Peso_Trayectoria_Escolar_Pct', ascending=False).head(15)
        
        if not df_pond.empty:
            # Crear etiqueta para el eje Y que incluya abreviación de la institución para diferenciar
            df_pond['Etiqueta_Y'] = df_pond['Nombre_Carrera'] + " (" + df_pond['Tipo_Institucion'].str[:3] + ")"
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=df_pond['Etiqueta_Y'], 
                x=df_pond['Peso_Trayectoria_Escolar_Pct'], 
                name='Notas Escolares (NEM+Ranking)', 
                orientation='h', 
                marker_color='#2ca02c'
            ))
            fig_bar.add_trace(go.Bar(
                y=df_pond['Etiqueta_Y'], 
                x=df_pond['Pond_PAES'], 
                name='Prueba PAES', 
                orientation='h', 
                marker_color='#1f77b4'
            ))
            fig_bar.update_layout(
                barmode='stack', 
                title="Top 15 Carreras que más premian la Trayectoria Escolar en la selección actual", 
                yaxis={'autorange': 'reversed'} # Mantiene el de mayor % NEM arriba
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PESTAÑA 4: JEFES UTP Y DIRECTIVOS
# ==========================================
with tab4:
    st.header("📊 Brechas de Mercado, Saturación y Tendencias")
    st.metric("Empleabilidad Promedio al 2do Año (Mercado Seleccionado)", f"{df_final['Empleabilidad_2do_Ano'].mean():.1f}%", delta="Meta >= 85%", delta_color="normal")
    st.markdown("---")
    
    # 1. Nueva Visualización: Análisis de Brechas y Saturación (Full Width para mejor lectura de áreas)
    st.subheader("Análisis de Brechas y Saturación por Área de Conocimiento")
    
    if len(df_final) > 0:
        # Agrupamos por área de conocimiento la sobredemanda
        df_saturacion = df_final.groupby('Area_Conocimiento')['Indice_Sobredemanda'].mean().reset_index()
        df_saturacion = df_saturacion.sort_values(by='Indice_Sobredemanda', ascending=False)
        
        fig_sat = px.bar(
            df_saturacion, 
            x='Area_Conocimiento', 
            y='Indice_Sobredemanda',
            color='Indice_Sobredemanda',
            color_continuous_scale='Reds',
            labels={"Area_Conocimiento": "Área de Conocimiento", "Indice_Sobredemanda": "Demanda por Vacante (x veces)"},
            title="Déficit y Saturación Académica (Valores > 1 indican más postulantes que cupos)"
        )
        # Línea de referencia estratégica (Equilibrio)
        fig_sat.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Equilibrio Oferta/Demanda (1.0)")
        fig_sat.update_layout(xaxis_tickangle=-45) # Inclina las etiquetas para que no se pisen
        st.plotly_chart(fig_sat, use_container_width=True)

    st.markdown("---")
    
    # Layout en columnas para las otras dos métricas
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Dispersión de Ingresos (Desigualdad Salarial)")
        fig_box = px.box(
            df_final, x="Tipo_Institucion", y="Ingreso_Promedio_4to_Ano", color="Tipo_Institucion",
            labels={"Ingreso_Promedio_4to_Ano": "Sueldo 4to Año", "Tipo_Institucion": "Subsistema"},
            template="plotly_white"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_m2:
        st.subheader("Matriz de Correlación Estratégica")
        cols_corr = ['Ingreso_Promedio_4to_Ano', 'Empleabilidad_1er_Ano', 'Duracion_Semestres', 'Arancel_Num']
        df_corr = df_final[cols_corr].dropna().corr()
        
        # Renombramos columnas temporalmente para que el heatmap sea más legible en la presentación
        df_corr.columns = ['Sueldo 4A', 'Empleabilidad', 'Semestres', 'Arancel']
        df_corr.index = ['Sueldo 4A', 'Empleabilidad', 'Semestres', 'Arancel']
        
        fig_heat = px.imshow(
            df_corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
