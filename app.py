import streamlit as st
import pandas as pd
import re
from googleapiclient.discovery import build
from openai import OpenAI


client = OpenAI(
    api_key=st.secrets["TOGETHER_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

youtube = build('youtube', 'v3', developerKey=st.secrets["YOUTUBE_API_KEY"])

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

def get_comments(video_id, max_results=1000):
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comment_date = item['snippet']['topLevelComment']['snippet']['publishedAt']
        comments.append({'comment': comment, 'date': comment_date})

    while 'nextPageToken' in response:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=response['nextPageToken'],
            maxResults=max_results
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment_date = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comments.append({'comment': comment, 'date': comment_date})

    return comments

st.set_page_config(page_title="YouTube Comments Analyzer", page_icon="🎬", layout="wide")
st.title("YouTube Comments AI Analyzer")
st.write("Enter a YouTube video URL to fetch comments and analyze them.")

video_url = st.text_input("YouTube Video URL")
max_comments = st.number_input("Max comments per request", min_value=10, max_value=1000, value=100, step=10)

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        comments_data = get_comments(video_id, max_results=max_comments)
        if comments_data:
            df = pd.DataFrame(comments_data)
            st.subheader("Comments Preview")
            st.dataframe(df.head(10))

            question = st.text_area(
                "What would you like to know about the comments?",
                "Summarize the overall sentiment, key topics, and viewer feedback."
            )

            if st.button("Analyze Comments"):
                with st.spinner("Analyzing comments... please wait"):
                    sample_comments = df["comment"].dropna().sample(min(100, len(df))).to_list()
                    comments_text = "\n".join(sample_comments)

                    prompt = f"""
You are a social media analyst AI.
Analyze these YouTube comments and answer the question below.

Comments:
{comments_text}

Question:
{question}
"""

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=1000
                    )

                    result = response.choices[0].message.content

                st.subheader("AI Analysis Result")
                st.write(result)
        else:
            st.warning("No comments found for this video.")
    else:
        st.error("Invalid YouTube URL.")
