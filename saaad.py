import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import numpy as np
import warnings
import os

# Supprimer les warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Voitures",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API Lambda
LAMBDA_URL = "https://w7e62hoex6.execute-api.us-east-1.amazonaws.com/prod/getScrapingData"

def clean_transmission(df):
    """Nettoie et normalise la colonne Transmission"""
    if 'Transmission' not in df.columns:
        return df
    
    # Convertir en minuscules et supprimer les espaces
    df['Transmission'] = df['Transmission'].astype(str).str.lower().str.strip()
    
    # Mapping des valeurs
    transmission_mapping = {
        'automatique': 'Automatique',
        'automatic': 'Automatique',
        'manuelle': 'Manuelle',
        'manual': 'Manuelle',
        'manuel': 'Manuelle',
        'diesel': None,  # Valeur incorrecte
        'essence': None,  # Valeur incorrecte
        'Diesel': None,  # Valeur incorrecte
        'Essence': None,  # Valeur incorrecte
        '-': None,
        'n/a': None,
        'nan': None
    }
    
    df['Transmission'] = df['Transmission'].map(transmission_mapping).fillna(df['Transmission'])
    
    # Mettre en majuscule la première lettre pour les valeurs non mappées valides
    df['Transmission'] = df['Transmission'].apply(
        lambda x: x.capitalize() if isinstance(x, str) and x not in ['nan', '-', 'n/a'] else None
    )
    
    return df

def clean_carburant(df):
    """Nettoie et normalise la colonne Carburant"""
    if 'Carburant' not in df.columns:
        return df
    
    # Convertir en minuscules et supprimer les espaces
    df['Carburant'] = df['Carburant'].astype(str).str.lower().str.strip()
    
    # Mapping des valeurs
    carburant_mapping = {
        'diesel': 'Diesel',
        'essence': 'Essence',
        'gasoline': 'Essence',
        'petrol': 'Essence',
        'hybride': 'Hybride',
        'hybrid': 'Hybride',
        'hybride-essence': 'Hybride Essence',
        'hybride plug-in': 'Hybride Rechargeable',
        'hybride rechargeable': 'Hybride Rechargeable',
        'electrique': 'Électrique',
        'eléctrique': 'Électrique',
        'électrique': 'Électrique',
        'diesel mhev': 'Diesel MHEV',
        'essence mhev': 'Essence MHEV',
        'n/a': None,
        '-': None,
        'nan': None,
        '6 cv': None,  # Valeur incorrecte
        '8 cv': None   # Valeur incorrecte
    }
    
    df['Carburant'] = df['Carburant'].map(carburant_mapping).fillna(df['Carburant'])
    
    # Nettoyer les valeurs non mappées
    df['Carburant'] = df['Carburant'].apply(
        lambda x: None if isinstance(x, str) and (x in ['nan', '-', 'n/a'] or 'cv' in x) else x
    )
    
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    """Charge les données depuis l'API Lambda"""
    try:
        response = requests.get(LAMBDA_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            return pd.DataFrame()

        for col in ['Km', 'Prix', 'Mc']:
            if col in df.columns:
                cleaned = df[col].astype(str).str.replace(r'[^0-9]', '', regex=True)
                cleaned = cleaned.replace('', pd.NA)
                df[col] = pd.to_numeric(cleaned, errors='coerce')

        if 'DateScraping' in df.columns:
            df['DateScraping'] = pd.to_datetime(df['DateScraping'], errors='coerce')

        columns_to_keep = [
            "Carburant",  "Cv", "DateScraping", "Etat",
            "Km", "location", "Marque", "Matricule", "Mc", "Modele",
            "Origine",  "Prix", "Source",
            "Statut", "Transmission", "Version"
        ]

        df = df[[c for c in columns_to_keep if c in df.columns]]
        df = df[
            df["Prix"].notna() &
            (df["Prix"] != 0)
        ]

        # Nettoyage de la colonne Transmission
        df = clean_transmission(df)
        
        # Nettoyage de la colonne Carburant
        df = clean_carburant(df)

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur lors de la récupération des données: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement des données: {e}")
        return pd.DataFrame()

def apply_filters(df):
    """Applique les filtres dépendants sélectionnés par l'utilisateur"""
    filtered_df = df.copy()
    
    st.sidebar.header("🔍 Filtres")
    
    # Filtre par Date de Scraping (premier filtre, indépendant)
    if 'DateScraping' in df.columns and not df['DateScraping'].isna().all():
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Filtrage par Date")
        
        date_level = st.sidebar.radio(
            "Niveau de filtrage:",
            ["Tous", "Année", "Mois", "Jour"],
            horizontal=True
        )
        
        if date_level == "Année":
            df_with_dates = filtered_df[filtered_df['DateScraping'].notna()].copy()
            if not df_with_dates.empty:
                annees = sorted(df_with_dates['DateScraping'].dt.year.unique().tolist())
                selected_annee = st.sidebar.selectbox("Choisir l'année", annees)
                filtered_df = filtered_df[
                    (filtered_df['DateScraping'].isna()) | 
                    (filtered_df['DateScraping'].dt.year == selected_annee)
                ]
        
        elif date_level == "Mois":
            df_with_dates = filtered_df[filtered_df['DateScraping'].notna()].copy()
            if not df_with_dates.empty:
                df_with_dates['AnneMois'] = df_with_dates['DateScraping'].dt.to_period('M').astype(str)
                mois_disponibles = sorted(df_with_dates['AnneMois'].unique().tolist())
                selected_mois = st.sidebar.selectbox("Choisir le mois", mois_disponibles)
                filtered_df = filtered_df[
                    (filtered_df['DateScraping'].isna()) | 
                    (filtered_df['DateScraping'].dt.to_period('M').astype(str) == selected_mois)
                ]
        
        elif date_level == "Jour":
            df_with_dates = filtered_df[filtered_df['DateScraping'].notna()].copy()
            if not df_with_dates.empty:
                jours_disponibles = sorted(df_with_dates['DateScraping'].dt.date.unique().tolist())
                selected_jour = st.sidebar.selectbox("Choisir le jour", jours_disponibles)
                filtered_df = filtered_df[
                    (filtered_df['DateScraping'].isna()) | 
                    (filtered_df['DateScraping'].dt.date == selected_jour)
                ]
    
    st.sidebar.markdown("---")
    
    # Filtre par Source (après la date)
    if 'Source' in df.columns:
        sources_disponibles = ['Tous'] + sorted(filtered_df['Source'].dropna().unique().tolist())
        selected_source = st.sidebar.selectbox("📍 Source", sources_disponibles)
        if selected_source != 'Tous':
            filtered_df = filtered_df[filtered_df['Source'] == selected_source]
    
    # Filtre par Marque (dépend de la date et source)
    if 'Marque' in df.columns:
        marques_disponibles = ['Tous'] + sorted(filtered_df['Marque'].dropna().unique().tolist())
        selected_marque = st.sidebar.selectbox("🏷️ Marque", marques_disponibles)
        if selected_marque != 'Tous':
            filtered_df = filtered_df[filtered_df['Marque'] == selected_marque]
    
    # Filtre par Modèle (dépend de la marque sélectionnée)
    if 'Modele' in df.columns:
        modeles_disponibles = ['Tous'] + sorted(filtered_df['Modele'].dropna().astype(str).unique().tolist())
        selected_modele = st.sidebar.selectbox("🚙 Modèle", modeles_disponibles)
        if selected_modele != 'Tous':
            filtered_df = filtered_df[filtered_df['Modele'].astype(str) == selected_modele]
    
    # Filtre par Cv (dépend de la marque et du modèle)
    if 'Cv' in df.columns:
        cvs_disponibles = ['Tous'] + sorted(filtered_df['Cv'].dropna().astype(str).unique().tolist())
        selected_cv = st.sidebar.selectbox("⚡ Chevaux (Cv)", cvs_disponibles)
        if selected_cv != 'Tous':
            filtered_df = filtered_df[filtered_df['Cv'].astype(str) == selected_cv]
    
    # Filtre par Carburant (dépend des filtres précédents)
    if 'Carburant' in df.columns:
        carburants_disponibles = ['Tous'] + sorted(filtered_df['Carburant'].dropna().unique().tolist())
        selected_carburant = st.sidebar.selectbox("⛽ Carburant", carburants_disponibles)
        if selected_carburant != 'Tous':
            filtered_df = filtered_df[filtered_df['Carburant'] == selected_carburant]
    
    # Filtre par Transmission (dépend des filtres précédents)
    if 'Transmission' in df.columns:
        transmissions_disponibles = ['Tous'] + sorted(filtered_df['Transmission'].dropna().unique().tolist())
        selected_transmission = st.sidebar.selectbox("⚙️ Transmission", transmissions_disponibles)
        if selected_transmission != 'Tous':
            filtered_df = filtered_df[filtered_df['Transmission'] == selected_transmission]
    
    # Filtre par Statut (dépend des filtres précédents)
    if 'Statut' in df.columns:
        statuts_disponibles = ['Tous'] + sorted(filtered_df['Statut'].dropna().unique().tolist())
        selected_statut = st.sidebar.selectbox("📊 Statut", statuts_disponibles)
        if selected_statut != 'Tous':
            filtered_df = filtered_df[filtered_df['Statut'] == selected_statut]
    
    # Filtre par État (dépend des filtres précédents)
    if 'Etat' in df.columns:
        etats_disponibles = ['Tous'] + sorted(filtered_df['Etat'].dropna().unique().tolist())
        selected_etat = st.sidebar.selectbox("🔧 État", etats_disponibles)
        if selected_etat != 'Tous':
            filtered_df = filtered_df[filtered_df['Etat'] == selected_etat]
    
    return filtered_df

def display_kpis(df):
    """Affiche les KPIs principaux"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚗 Total Voitures", len(df))
    
    with col2:
        if 'Prix' in df.columns and not df['Prix'].isna().all():
            prix_moyen = df['Prix'].mean()
            st.metric("💰 Prix Moyen", f"{prix_moyen:,.0f} MAD" if not pd.isna(prix_moyen) else "N/A")
    
    with col3:
        if 'Km' in df.columns and not df['Km'].isna().all():
            km_moyen = df['Km'].mean()
            st.metric("📊 KM Moyen", f"{km_moyen:,.0f}" if not pd.isna(km_moyen) else "N/A")
    
    with col4:
        if 'Source' in df.columns:
            nb_sources = df['Source'].nunique()
            st.metric("📍 Sources", nb_sources)

def display_charts(df):
    """Affiche tous les graphiques"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Source' in df.columns and not df['Source'].isna().all():
            st.subheader("📍 Répartition par Source")
            df_source = df[df['Source'].notna()].copy()
            if not df_source.empty:
                fig = px.histogram(
                    df_source,
                    x='Source',
                    title="Nombre de voitures par Source",
                    color='Source'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Transmission' in df.columns and not df['Transmission'].isna().all():
            st.subheader("⚙️ Répartition par Transmission")
            df_trans = df[df['Transmission'].notna()].copy()
            if not df_trans.empty:
                fig = px.histogram(
                    df_trans, 
                    x='Transmission', 
                    title="Nombre de voitures par transmission",
                    color='Transmission'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        if 'Carburant' in df.columns and not df['Carburant'].isna().all():
            st.subheader("⛽ Répartition par Carburant")
            df_carburant = df[df['Carburant'].notna()].copy()
            if not df_carburant.empty:
                fig = px.histogram(
                    df_carburant, 
                    x='Carburant', 
                    title="Nombre de voitures par carburant",
                    color='Carburant'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        if 'Etat' in df.columns and not df['Etat'].isna().all():
            st.subheader("🔧 Répartition par État")
            df_etat = df[df['Etat'].notna()].copy()
            if not df_etat.empty:
                fig = px.histogram(
                    df_etat, 
                    x='Etat', 
                    title="Nombre de voitures par état",
                    color='Etat'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    if 'Marque' in df.columns and not df['Marque'].isna().all():
        st.subheader("🏷️ Top 10 des Marques")
        df_marque = df[df['Marque'].notna()].copy()
        if not df_marque.empty:
            top_marques = df_marque['Marque'].value_counts().head(10)
            fig = px.bar(
                x=top_marques.index,
                y=top_marques.values,
                title="Top 10 des marques les plus représentées",
                labels={'x': 'Marque', 'y': 'Nombre de voitures'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    if 'DateScraping' in df.columns and not df['DateScraping'].isna().all():
        st.subheader("📅 Évolution temporelle")
        df_date = df[df['DateScraping'].notna()].copy()
        if not df_date.empty:
            date_counts = df_date.groupby(df_date['DateScraping'].dt.date).size().reset_index()
            date_counts.columns = ['Date', 'Nombre']
            fig = px.line(
                date_counts,
                x='Date',
                y='Nombre',
                title="Évolution du nombre de voitures scrapées par date",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

def display_statistics(df):
    """Affiche les statistiques détaillées"""
    st.subheader("📊 Statistiques Détaillées")
    
    numerical_cols = ['Km', 'Prix', 'Mc']
    available_cols = [col for col in numerical_cols if col in df.columns and not df[col].isna().all()]
    
    if available_cols:
        stats_df = pd.DataFrame()
        
        for col in available_cols:
            stats_df[col] = [
                df[col].mean(),
                df[col].median(),
                df[col].std(),
                df[col].min(),
                df[col].max(),
                df[col].count()
            ]
        
        stats_df.index = ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Nombre de valeurs']
        stats_df = stats_df.round(2)
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("Aucune donnée numérique disponible pour les statistiques.")

def main():
    st.title("🚗 Dashboard Voitures")
    st.markdown("---")
    
    with st.spinner("🔄 Chargement des données en cours..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Aucune donnée disponible.")
        st.stop()
    
    st.success(f"✅ {len(df)} voitures chargées avec succès")
    
    filtered_df = apply_filters(df)
    
    if len(filtered_df) != len(df):
        st.info(f"🔍 {len(filtered_df)} voitures correspondent aux filtres sélectionnés (sur {len(df)} total)")
    
    display_kpis(filtered_df)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 Graphiques", "📊 Statistiques", "🗂️ Données"])
    
    with tab1:
        display_charts(filtered_df)
    
    with tab2:
        display_statistics(filtered_df)
        
        st.subheader("🏷️ Analyse par Catégories")
        categorical_cols = ['Source', 'Etat', 'Transmission', 'Carburant', 'Statut', 'Marque']
        available_cat_cols = [col for col in categorical_cols if col in filtered_df.columns]
        
        for col in available_cat_cols:
            if not filtered_df[col].isna().all():
                st.write(f"**{col}:**")
                category_counts = filtered_df[col].value_counts()
                st.dataframe(category_counts.to_frame('Nombre'), use_container_width=True)
                st.markdown("---")
    
    with tab3:
        st.subheader("🗂️ Données Brutes")
        st.dataframe(filtered_df, use_container_width=True)
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"voitures_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informations")
    st.sidebar.info(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    if st.sidebar.button("🔄 Actualiser les données"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.subheader("created by ghizlane chtouki")

if __name__ == "__main__":
    main()