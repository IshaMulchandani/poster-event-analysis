# I have a python assessment for which I have to build a streamlit app. I must pretend as though I am part of the web development team of the National Poster presentation Event.
# This webpage will have the following features:
# 1. Dataset generation: I must use python libraries to generate a dataset which has 10 columns and 450 rows. These columns should be a mix of categorical and numerical data.The dataset generated through code will have to be saved as a .csv file. The dataset should include the following columns:
# - Student ID: Unique identifier for each student (numerical)
# - Names: Names of students
# - College: College names
# - Day: each student presented their poster on 1 out of 4 days for which the event ran
# - Poster Title: Title of the poster presented by the student
# - Track: The track under which the poster was presented (Only 4 tracks were there in the whole event)
# - Student Email: Email of the student
# - Student Phone: Phone number of the student
# - State: State of the student (Indian States only)
# - Feedback : a long feedback text given by the student about the event

# 2. Dashboard Development: I must create a dashboard using streamlit that will display the following:
# - A title for the dashboard
# - A single dynamic line chart that allows users to see the number of posters presented classified by:
#     - By college
#     - By track
#     - By poster type
#     - By state
# - A download button that allows users to download the dataset as a CSV file

# 3. Text Analysis: For this feature, I have to analyse the feedback text provided by the students on completeion of the event. I have to execute the following tasks:
# - Generate a wordcloud which is separate for each track in the event across all days
# - Implement a text similarity algorithm like jaccard similarity to show similarity within feedbacks given by students separately within each track.

# 4. Image processing: This module is for track related photos. It should have the following features:
# - Day-wise image gallery
# - Image upload option for each day
# - Image processing using OpenCV to apply filters on the uploaded images. The user should be able to select what kind of filter or alterations they want to apply on the images. The filters should include:
#     - Grayscale
#     - Gaussian Blur
#     - Median Blur
#     - Color conversion
#     - Edge detection
# - The processed images should be displayed in the app.
# - A download button for the processed images
    
# General Instructions:
# 1. The app should be user friendly
# 2. The app should be aesthetic and visually appealing
# 3. The app should use best coding practices to optimize the output.
# 4. The app should use maodern designing methods to make the app look good.
# 5. The app should use advanced streamlit widgets and elements to make it interactive.
# 6. The entire code for the app should be in one file titled app.py.
# 7. If any assets are required, you should flag them within the code itself and then at the end of code generation you should make a list of all images or other assets I need to download and set up in my working directory.

# Before proceeding with code generation, you must give me an outline of how you will design this app, what modules it will contain, the color palette you will use for the website and all the libraries you will be using. After getting my approval on the rough structure, you will proceed with code generation. Read the instructions above carefully, if I have missed out any details or anything is unclear to you, forst ask me questions and clarify all your doubts before proceeding. You are to ask as many questions as you need but during the development process you will make no assumptions.

import streamlit as st
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from faker import Faker
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import cv2
import os
import hashlib

# Set Streamlit Page Configuration (MUST BE FIRST Streamlit command)
st.set_page_config(page_title="National Poster Presentation Event", layout="wide")

# Initialize Faker
fake = Faker()

# Function to generate a dataset
def generate_dataset():
    tracks = ["Artificial Intelligence", "Blockchain", "Cybersecurity", "IoT"]
    colleges = ["IIT Bombay", "NIT Trichy", "BITS Pilani", "Anna University", "Delhi University"]
    states = ["Maharashtra", "Tamil Nadu", "Karnataka", "Delhi", "West Bengal"]
    event_days = ["Day 1", "Day 2", "Day 3", "Day 4"]

    data = {
        "Student ID": np.arange(1001, 1451),
        "Names": [fake.name() for _ in range(450)],
        "College": [random.choice(colleges) for _ in range(450)],
        "Day": [random.choice(event_days) for _ in range(450)],
        "Poster Title": [" ".join(fake.words(nb=4)).title() for _ in range(450)],
        "Track": [random.choice(tracks) for _ in range(450)],
        "Student Email": [fake.email() for _ in range(450)],
        "Student Phone": [fake.phone_number()[:10] for _ in range(450)],
        "State": [random.choice(states) for _ in range(450)],
        "Feedback": [fake.sentence(nb_words=15) for _ in range(450)]
    }

    return pd.DataFrame(data)

# Store dataset in session state for persistence
if "df" not in st.session_state:
    st.session_state.df = generate_dataset()

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Dashboard", "Text Analysis", "Image Processing"])

# Dataset Regeneration Button (Updates in session_state)
if st.sidebar.button("Regenerate Dataset"):
    st.session_state.df = generate_dataset()
    st.sidebar.success("Dataset Regenerated!")

# Fetch the updated dataset
df = st.session_state.df

# Dashboard Section
if option == "Dashboard":
    st.header("📊 Event Dashboard")

    # Select parameter for visualization
    filter_option = st.selectbox("Select parameter for visualization:", ["State", "College", "Track"])

    # Line Chart: Posters Presented Over Time
    fig1 = px.line(df.groupby(filter_option)["Poster Title"].count().reset_index(),
                   x=filter_option, y="Poster Title", title=f"Posters Presented by {filter_option}")
    st.plotly_chart(fig1, use_container_width=True)

    # Bar Chart: Number of Posters per Category
    bar_data = df[filter_option].value_counts().reset_index()
    bar_data.columns = [filter_option, "Count"]  # Rename columns properly

    fig2 = px.bar(bar_data, x=filter_option, y="Count", title=f"Poster Count by {filter_option}",
                  labels={filter_option: filter_option, "Count": "Number of Posters"},
                  color=filter_option)
    st.plotly_chart(fig2, use_container_width=True)

    # Pie Chart: Distribution of Posters
    fig3 = px.pie(df, names=filter_option, title=f"Poster Distribution by {filter_option}",
                  hole=0.3, color=filter_option)
    st.plotly_chart(fig3, use_container_width=True)

    # Box Plot: Poster Count Distribution
    box_data = df.groupby(filter_option)["Poster Title"].count().reset_index()
    box_data.columns = [filter_option, "Poster Count"]

    fig4 = px.box(box_data, x=filter_option, y="Poster Count", title=f"Distribution of Posters by {filter_option}",
                  points="all")
    st.plotly_chart(fig4, use_container_width=True)

    # Heatmap: Correlation Between Categorical Features
    pivot_table = df.pivot_table(index="State", columns="Track", values="Poster Title", aggfunc="count", fill_value=0)
    fig5, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt="d", linewidths=0.5, ax=ax)
    st.pyplot(fig5)

    # Dataset Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Dataset", data=csv, file_name="event_dataset.csv", mime="text/csv")



# Text Analysis Section
elif option == "Text Analysis":
    st.header("📝 Text Analysis")

    # WordCloud per track
    track_selected = st.selectbox("Select a track for WordCloud", df["Track"].unique())

    feedback_text = " ".join(df[df["Track"] == track_selected]["Feedback"])
    
    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(feedback_text)

    # Display WordCloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Function to calculate Jaccard Similarity per track
    def calculate_jaccard_similarity(track):
        track_feedbacks = df[df["Track"] == track]["Feedback"].tolist()
        vectorizer = CountVectorizer(binary=True).fit(track_feedbacks)
        vectors = vectorizer.transform(track_feedbacks).toarray()

        if len(vectors) > 1:
            similarity_scores = [jaccard_score(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
            return round(np.mean(similarity_scores), 2)
        return 0.0

    # Compute and display Jaccard Similarity below WordCloud
    similarity_score = calculate_jaccard_similarity(track_selected)
    st.markdown(f"### 🔹 Jaccard Similarity for *{track_selected}*: `{similarity_score}`")


# Image Processing Section
elif option == "Image Processing":
    st.header("🖼️ Day-wise Albums")

    # Event Days Selection
    event_days = ["Day 1", "Day 2", "Day 3", "Day 4"]
    selected_day = st.selectbox("Select Event Day", event_days)

    # Initialize session state for storing images per day
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = {day: {} for day in event_days}  # Dictionary for unique storage

    # Upload Image for Selected Day
    uploaded_file = st.file_uploader(f"Upload an image for {selected_day}", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Convert image to correct color format (BGR to RGB)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate a unique hash for the image to prevent duplicates
        image_hash = hashlib.md5(image.tobytes()).hexdigest()

        if image_hash not in st.session_state.uploaded_images[selected_day]:
            st.session_state.uploaded_images[selected_day][image_hash] = image  # Store image by hash

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Display thumbnails of previously uploaded images
    st.subheader(f"📂 {selected_day} Album")
    if len(st.session_state.uploaded_images[selected_day]) > 0:
        col1, col2, col3 = st.columns(3)  # Create columns for thumbnails

        image_keys = list(st.session_state.uploaded_images[selected_day].keys())  # Get all stored image hashes
        selected_key = st.radio("Select an image", image_keys, format_func=lambda k: f"Image {image_keys.index(k) + 1}")

        # Display thumbnails
        for i, key in enumerate(image_keys):
            with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
                st.image(st.session_state.uploaded_images[selected_day][key], caption=f"Image {i+1}", use_container_width=True)

        # Display selected image
        selected_image = st.session_state.uploaded_images[selected_day][selected_key]
        st.image(selected_image, caption=f"Viewing Image {image_keys.index(selected_key) + 1}", use_container_width=True)

        # Image Processing Module
        st.subheader("🛠️ Apply Filters")
        filter_option = st.selectbox("Choose a filter", ["Original", "Grayscale", "Gaussian Blur", "Median Blur", "Color Conversion", "Edge Detection"])

        # Apply filters
        if filter_option == "Original":
            processed_image = selected_image
        elif filter_option == "Grayscale":
            processed_image = cv2.cvtColor(selected_image, cv2.COLOR_RGB2GRAY)
        elif filter_option == "Gaussian Blur":
            processed_image = cv2.GaussianBlur(selected_image, (15, 15), 0)
        elif filter_option == "Median Blur":
            processed_image = cv2.medianBlur(selected_image, 5)
        elif filter_option == "Color Conversion":
            processed_image = cv2.cvtColor(selected_image, cv2.COLOR_RGB2HSV)
        elif filter_option == "Edge Detection":
            processed_image = cv2.Canny(selected_image, 100, 200)

        # Display processed image
        st.image(processed_image, caption=f"Processed Image - {filter_option}", use_container_width=True)

        # Download processed image
        _, processed_image_bytes = cv2.imencode(".png", processed_image)
        st.download_button(label="Download Processed Image", data=processed_image_bytes.tobytes(),
                           file_name="processed_image.png", mime="image/png")

    else:
        st.info("No images uploaded for this day yet.")
