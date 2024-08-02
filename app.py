import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import chardet
import os

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    labels = ['NEGATIVE', 'POSITIVE']
    score, label = torch.max(scores, dim=1)
    return {"label": labels[label.item()], "score": score.item()}

# Function to detect file encoding
def detect_encoding(file):
    rawdata = file.read()
    result = chardet.detect(rawdata)
    return result['encoding']

def generate_pdf(pie_chart_path, pos_wordcloud_path, neg_wordcloud_path):
    pdf_output = BytesIO()
    pdf_height = 16.5 * inch  # Total vertical height calculated
    pdf_width = 8.27 * inch  # A4 width
    c = canvas.Canvas(pdf_output, pagesize=(pdf_width, pdf_height))

    # Set starting vertical position
    y_position = pdf_height - 1 * inch

    # Add title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(2.2 * inch, y_position, "Sentiment Analysis Report")

    # Update vertical position after title
    y_position -= 2 * inch

    # Add pie chart with width 5 inches and height double the width
    pie_chart_width = 5 * inch
    pie_chart_height = 4 * inch
    c.drawImage(pie_chart_path, 1.5 * inch, y_position - pie_chart_height, width=pie_chart_width, height=pie_chart_height)

    # Update vertical position after pie chart
    y_position -= (pie_chart_height + 1 * inch)  # Add some spacing

    # Add Positive Keywords heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(3 * inch, y_position, "Positive Keywords")

    # Add positive word cloud
    c.drawImage(pos_wordcloud_path, 1 * inch, y_position - 3.3 * inch, width=6 * inch, height=3 * inch)  # 2:1 ratio

    # Update vertical position after positive word cloud
    y_position -= (3 * inch + 1 * inch)  # Add some spacing

    # Add Negative Keywords heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(3 * inch, y_position, "Negative Keywords")

    # Add negative word cloud
    c.drawImage(neg_wordcloud_path, 1 * inch, y_position - 3.3 * inch, width=6 * inch, height=3 * inch)  # 2:1 ratio

    c.save()
    pdf_output.seek(0)

    return pdf_output


# Streamlit UI
st.title("Sentiment Analysis and Reporting")

# Initialize session state for button visibility
if 'show_pdf_download' not in st.session_state:
    st.session_state.show_pdf_download = False

# Sidebar for encoding detection and reset button
st.sidebar.header("File Encoding Checker")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV file for Encoding Check", type=["csv"])

if uploaded_file:
    # Detect the encoding
    encoding = detect_encoding(uploaded_file)
    st.sidebar.write(f"Detected encoding: {encoding}")

# Reset button in the sidebar
if st.sidebar.button("Reset Analysis"):
    if os.path.exists("sentiment_pie_chart.png"):
        os.remove("sentiment_pie_chart.png")
    if os.path.exists("pos_wordcloud.png"):
        os.remove("pos_wordcloud.png")
    if os.path.exists("neg_wordcloud.png"):
        os.remove("neg_wordcloud.png")
    st.sidebar.write("Files deleted. Please re-upload a file to start over.")

# File uploader for sentiment analysis
uploaded_file = st.file_uploader("Upload CSV file for Sentiment Analysis", type=["csv"])

# Dropdown for encoding specification in the main panel
encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'ASCII', 'UTF-16', 'UTF-32', 'ANSI', "Windows-1251", 'Windows-1252']
user_encoding = st.selectbox("Select Encoding", options=encodings, index=0)

# Button to start processing
if st.button("Go"):
    if uploaded_file:
        try:
            # Load the CSV file into DataFrame with specified encoding
            uploaded_file.seek(0)  # Reset the file pointer to the beginning
            df = pd.read_csv(uploaded_file, encoding=user_encoding)
        except UnicodeDecodeError:
            st.error("Error decoding the file. Please specify the correct encoding.")
        else:
            # Check if the DataFrame has exactly one column
            if df.shape[1] != 1:
                st.warning("The CSV file should only contain one column with review data.")
            else:
                # Rename the column to 'review'
                df.columns = ['review']

                # Clean up the DataFrame
                df['review'] = df['review'].astype(str).str.strip()
                df = df[df['review'].apply(len) <= 512]

                # Apply sentiment analysis
                df['sentiment'] = df['review'].apply(analyze_sentiment)
                df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'])
                df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'])

                # Drop the original 'sentiment' column
                df = df.drop(columns=['sentiment'])

                # Pie chart data
                sentiment_counts = df['sentiment_label'].value_counts()

                # Create pie chart
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=45)
                ax.set_title('Distribution of Sentiment')
                pie_chart_path = "sentiment_pie_chart.png"
                plt.savefig(pie_chart_path)

                # Create word clouds
                stopwords = set(STOPWORDS)

                pos_reviews = df[df['sentiment_label'] == 'POSITIVE']['review'].str.cat(sep=' ')
                neg_reviews = df[df['sentiment_label'] == 'NEGATIVE']['review'].str.cat(sep=' ')

                pos_wordcloud = WordCloud(max_font_size=80, max_words=10, background_color='white', stopwords=stopwords).generate(pos_reviews)
                neg_wordcloud = WordCloud(max_font_size=80, max_words=10, background_color='white', stopwords=stopwords).generate(neg_reviews)

                # Save word clouds to files
                pos_wordcloud_path = "pos_wordcloud.png"
                neg_wordcloud_path = "neg_wordcloud.png"
                pos_wordcloud.to_file(pos_wordcloud_path)
                neg_wordcloud.to_file(neg_wordcloud_path)

                # Create PDF
                pdf_output = generate_pdf(pie_chart_path, pos_wordcloud_path, neg_wordcloud_path)

                # Display options
                st.write("Processing complete!")

                # Update session state to show the appropriate buttons
                st.session_state.show_pdf_download = True

                # Display buttons
                download_pdf = st.download_button("Download PDF Report", pdf_output, file_name="sentiment_analysis_report.pdf", mime="application/pdf")
    else:
        st.info("Please upload a CSV file to get started.")
