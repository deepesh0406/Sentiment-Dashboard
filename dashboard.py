import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
from textblob import TextBlob
from transformers import pipeline

# ------------------- Page Configuration -------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Real-Time Sentiment & Emotion Dashboard</h1>", unsafe_allow_html=True)

# ------------------- Load Emotion Model -------------------
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_model = load_emotion_model()

# ------------------- Sentiment & Emotion Functions -------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def analyze_emotion(text):
    try:
        result = emotion_model(text[:512])[0]
        top_emotion = max(result, key=lambda x: x["score"])
        return top_emotion["label"]
    except:
        return "Unknown"

# ------------------- User Feedback Input Section -------------------
st.markdown("###  Submit Feedback for Analysis")
with st.form("feedback_form"):
    user_input = st.text_area("Write your comment, feedback, or opinion here...", height=100)
    submitted = st.form_submit_button("Analyze")

if submitted and user_input:
    sentiment = analyze_sentiment(user_input)
    emotion = analyze_emotion(user_input)
    st.success("▶ Analysis Complete!")
    st.markdown(f"**Sentiment:** `{sentiment}`")
    st.markdown(f"**Emotion:** `{emotion}`")

# ------------------- File Upload or Default -------------------
uploaded_file = st.sidebar.file_uploader(" Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
   df = pd.read_csv("data/test_with_predictions_and_emotions.csv")


df.columns = df.columns.str.strip()

# ------------------- Sidebar Filters -------------------
st.sidebar.header(" Filter Data")
country_filter = st.sidebar.selectbox(" Select Country", ["All"] + sorted(df["Country"].dropna().unique()))
age_filter = st.sidebar.selectbox(" Select Age Group", ["All"] + sorted(df["Age of User"].dropna().unique()))
emotion_filter = st.sidebar.selectbox(" Select Emotion", ["All"] + sorted(df["Detected_Emotion"].dropna().unique()) if "Detected_Emotion" in df.columns else ["All"])

if country_filter != "All":
    df = df[df["Country"] == country_filter]
if age_filter != "All":
    df = df[df["Age of User"] == age_filter]
if "Detected_Emotion" in df.columns and emotion_filter != "All":
    df = df[df["Detected_Emotion"] == emotion_filter]

# ------------------- Preview Data -------------------
st.markdown("###  Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ------------------- Tabbed View -------------------
tab1, tab2 = st.tabs([" Sentiment Analysis", " Emotion Detection"])

# ------------------- TAB 1: SENTIMENT -------------------
with tab1:
    st.markdown("###  Sentiment Distribution")
    sentiment_counts = df["Predicted_Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig1 = px.pie(sentiment_counts, names='Sentiment', values='Count', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig1, use_container_width=True)

    # Sentiment trend over time
    if "Time of Tweet" in df.columns:
        st.markdown("###  Sentiment Trend Over Time")
        time_data = df.copy()
        time_data['Time of Tweet'] = pd.to_datetime(time_data['Time of Tweet'], format="%d/%m/%Y %H:%M", errors='coerce')
        time_data = time_data.dropna(subset=['Time of Tweet'])

        if not time_data.empty:
            time_data['Hour'] = time_data['Time of Tweet'].dt.hour
            chart_data = time_data.groupby(['Hour', 'Predicted_Sentiment']).size().reset_index(name='Count')

            if not chart_data.empty:
                alt_chart = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X('Hour:O', title='Hour of Day'),
                    y=alt.Y('Count:Q', title='Tweet Count'),
                    color=alt.Color('Predicted_Sentiment:N', title='Sentiment'),
                    tooltip=['Hour', 'Predicted_Sentiment', 'Count']
                ).properties(title=" Sentiment Trend Over Time")
                st.altair_chart(alt_chart, use_container_width=True)
            else:
                st.warning("Not enough data to show sentiment trend.")
        else:
            st.warning("No valid timestamps found to plot sentiment trend.")

    # Sentiment by Country
    if "Country" in df.columns:
        st.markdown("###  Sentiment by Country")
        fig3 = px.histogram(df, x="Country", color="Predicted_Sentiment", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)
            # ------------------- Positive Sentiment Map -------------------
    st.markdown("### 🌎 Global Map: Positive Sentiment by Country")
    try:
        map_df = df[df["Predicted_Sentiment"].str.lower() == "positive"]
        country_counts = map_df["Country"].value_counts().reset_index()
        country_counts.columns = ["Country", "Positive_Count"]

        if not country_counts.empty:
            fig_map = px.choropleth(
                country_counts,
                locations="Country",
                locationmode="country names",
                color="Positive_Count",
                color_continuous_scale="Greens",
                title="Positive Sentiment Across Countries"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No positive sentiment data available for mapping.")
    except Exception as e:
        st.warning("🌐 Could not generate map. Check country data format.")


# ------------------- TAB 2: EMOTION -------------------
with tab2:
    st.markdown("###  Emotion Distribution")
    if "Detected_Emotion" in df.columns:
        emotion_counts = df["Detected_Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        fig4 = px.bar(emotion_counts, x="Emotion", y="Count", color="Emotion", 
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No emotion data available.")

# ------------------- Word Cloud -------------------
st.markdown("###  Word Cloud from Tweets")
if "Cleaned_Text" in df.columns:
    text_data = " ".join(df["Cleaned_Text"].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    fig5, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig5)

# ------------------- Export Button -------------------
st.markdown("###  Download Filtered Data")
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

csv_file = convert_df_to_csv(df)
st.download_button(label=" Download as CSV", data=csv_file, file_name="filtered_sentiment_data.csv", mime="text/csv")
