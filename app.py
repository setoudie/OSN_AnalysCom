import streamlit as st
import pandas as pd
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
from textblob_fr import PatternAnalyzer
import io


# Charger le modèle spaCy XXXX 
@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_md")


nlp = load_spacy_model()


# Fonction d'extraction des mots-clés
def extract_keywords(comment, num_keywords=4):
    if not comment or pd.isna(comment):
        return []

    doc = nlp(str(comment))
    keywords = []

    # Liste personnalisée de stopwords français
    custom_stopwords = {"avis", "client", "produit", "service", "entreprise", "société"}

    for token in doc:
        if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and
                not token.is_stop and
                not token.is_punct and
                len(token.lemma_) > 2 and
                token.lemma_.lower() not in custom_stopwords):
            keywords.append(token.lemma_.lower())

    keyword_counts = Counter(keywords)
    return [word for word, _ in keyword_counts.most_common(num_keywords)]


# Analyse de sentiment améliorée
def enhanced_sentiment_analysis(comment, keywords):
    blob = TextBlob(str(comment), analyzer=PatternAnalyzer())
    polarity = blob.sentiment[0]

    # Dictionnaire étendu de mots à forte polarité
    POSITIVE_WORDS = {"excellent", "super", "parfait", "recommandé", "génial", "satisfait",
                      "efficace", "rapide", "professionnel", "agréable"}
    NEGATIVE_WORDS = {"déçu", "mauvais", "horrible", "éviter", "problème", "défectueux",
                      "lent", "cher", "compliqué", "déception"}

    for word in keywords:
        if word in POSITIVE_WORDS:
            polarity = min(1.0, polarity + 0.15)
        elif word in NEGATIVE_WORDS:
            polarity = max(-1.0, polarity - 0.15)

    if polarity > 0.2:
        return "Positif", polarity
    elif polarity < -0.2:
        return "Négatif", polarity
    else:
        return "Neutre", polarity


# Interface Streamlit
st.set_page_config(page_title="Analyse de Commentaires", layout="wide")
st.title("📊 Analyse Optimisée des Commentaires Clients")

# Chargement des données
uploaded_file = st.file_uploader("Télécharger un fichier de commentaires (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Vérifier la présence de la colonne "Zone"
    if 'Zone' not in df.columns:
        df['Zone'] = "Non spécifiée"

    # Traitement des commentaires
    with st.spinner("Analyse des commentaires en cours..."):
        df['Mots_cles'] = df['Commentaire'].apply(extract_keywords)
        df[['Sentiment', 'Polarite']] = df.apply(
            lambda row: enhanced_sentiment_analysis(row['Commentaire'], row['Mots_cles']),
            axis=1, result_type='expand'
        )

    # ===== FILTRES =====
    st.sidebar.header("Filtres")

    # Filtre par zone
    zones = ["Toutes les zones"] + list(df['Zone'].unique())
    selected_zone = st.sidebar.selectbox("Zone géographique", zones)

    # Filtre par sentiment
    sentiments = ["Tous"] + list(df['Sentiment'].unique())
    selected_sentiment = st.sidebar.selectbox("Type de sentiment", sentiments)

    # Appliquer les filtres
    filtered_df = df.copy()

    if selected_zone != "Toutes les zones":
        filtered_df = filtered_df[filtered_df['Zone'] == selected_zone]

    if selected_sentiment != "Tous":
        filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]

    # ===== TÉLÉCHARGEMENT =====
    st.sidebar.header("Exporter les résultats")

    # Format d'export
    export_format = st.sidebar.radio("Format d'export", ["CSV", "Excel"])


    # Fonction de conversion pour le téléchargement
    def convert_df_to_csv(df):
        return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Commentaires')
        return output.getvalue()


    # Bouton de téléchargement
    if export_format == "CSV":
        st.sidebar.download_button(
            label="Télécharger les résultats (CSV)",
            data=convert_df_to_csv(filtered_df),
            file_name='commentaires_analyses.csv',
            mime='text/csv'
        )
    else:
        st.sidebar.download_button(
            label="Télécharger les résultats (Excel)",
            data=convert_df_to_excel(filtered_df),
            file_name='commentaires_analyses.xlsx',
            mime='application/vnd.ms-excel'
        )

    # ===== AFFICHAGE DES RÉSULTATS =====
    # KPI en haut de page
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total commentaires", len(filtered_df))
    with col2:
        pos_count = len(filtered_df[filtered_df['Sentiment'] == 'Positif'])
        st.metric("Commentaires positifs", f"{pos_count} ({pos_count / len(filtered_df) * 100:.1f}%)")
    with col3:
        neg_count = len(filtered_df[filtered_df['Sentiment'] == 'Négatif'])
        st.metric("Commentaires négatifs", f"{neg_count} ({neg_count / len(filtered_df) * 100:.1f}%)")

    # Tableau des résultats
    st.subheader("Commentaires analysés")
    st.dataframe(filtered_df[['Commentaire', 'Zone', 'Mots_cles', 'Sentiment', 'Polarite']].head(20),
                 height=600)

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Répartition des sentiments")
        sentiment_counts = filtered_df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    with col2:
        st.subheader("Répartition par zone")
        zone_counts = filtered_df['Zone'].value_counts()
        st.bar_chart(zone_counts)

    # Nuage de mots-clés
    st.subheader("Nuage de mots-clés")

    if len(filtered_df) > 0:
        all_keywords = [word for sublist in filtered_df['Mots_cles'] for word in sublist]

        if all_keywords:
            keyword_freq = Counter(all_keywords)

            # Création du wordcloud avec des paramètres améliorés
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                colormap='viridis',
                max_words=50,
                max_font_size=150,
                collocations=False
            ).generate_from_frequencies(keyword_freq)

            # Affichage
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # Top 20 des mots-clés
            st.caption("Top 20 des mots-clés les plus fréquents")
            top_keywords = pd.DataFrame(keyword_freq.most_common(20), columns=['Mot-clé', 'Fréquence'])
            st.bar_chart(top_keywords.set_index('Mot-clé'))
        else:
            st.warning("Aucun mot-clé détecté dans les commentaires filtrés.")
    else:
        st.warning("Aucun commentaire à afficher avec les filtres sélectionnés.")

    # Analyse détaillée par zone
    st.subheader("Analyse par zone")

    if len(filtered_df['Zone'].unique()) > 1:
        zone_analysis = filtered_df.groupby('Zone')['Sentiment'].value_counts(normalize=True).unstack()
        zone_analysis = zone_analysis.fillna(0)
        zone_analysis = zone_analysis[['Positif', 'Neutre', 'Négatif']] * 100

        st.dataframe(zone_analysis.style.format("{:.1f}%").background_gradient(cmap='Blues'))
    else:
        st.info("Sélectionnez plusieurs zones pour comparer les performances")

# Instructions
with st.expander("ℹ️ Guide d'utilisation"):
    st.markdown("""
    **Comment utiliser cette application :**
    1. Téléchargez un fichier CSV contenant des commentaires clients
    2. Utilisez les filtres dans la barre latérale pour affiner l'analyse
    3. Explorez les résultats dans les différentes sections
    4. Téléchargez les résultats analysés au format CSV ou Excel

    **Format du fichier attendu :**
    - Colonne obligatoire : `Commentaire` (texte des commentaires)
    - Colonne optionnelle : `Zone` (pour le filtrage géographique)
    - Toutes les autres colonnes seront conservées mais non utilisées dans l'analyse

    **Optimisations incluses :**
    - Extraction des 4 mots-clés les plus pertinents par commentaire
    - Analyse de sentiment améliorée avec dictionnaire personnalisé
    - Nuage de mots basé uniquement sur les mots-clés extraits
    - Filtres par zone géographique et par type de sentiment
    - Téléchargement des résultats au format CSV ou Excel
    """)

st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ by Seny | v1.0")