import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import calmap
import networkx as nx
from collections import Counter
import emoji
import re
from urlextract import URLExtract


# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Helper functions (merged from helper.py)
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    extract = URLExtract()
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_active_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

def most_common_word(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " - " + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract the day name (Monday, Tuesday, etc.)
    df['day_name'] = df['date'].dt.day_name()

    # Extract the hour in AM/PM format
    df['hour'] = df['date'].dt.strftime('%I %p')  # Converts hour to '01 AM', '02 PM', etc.

    # Filter by user if needed
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Create the pivot table (heatmap) grouped by day of the week and hourly period
    user_heatmap = df.pivot_table(index='day_name', columns='hour', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def generate_report(selected_user, df):
    # Create a markdown report with analysis insights
    report = f"# WhatsApp Chat Analytics Report\n\n"
    report += f"## Analysis for User: {selected_user}\n\n"

    # Basic stats
    stats = fetch_stats(selected_user, df)
    report += "### Message Statistics\n"
    report += f"- Total Messages: {stats[0]}\n"
    report += f"- Total Words: {stats[1]}\n"
    report += f"- Media Shared: {stats[2]}\n"
    report += f"- Links Shared: {stats[3]}\n\n"

    # Sentiment overview
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    avg_sentiment = df['sentiment'].mean()
    report += "### Sentiment Overview\n"
    report += f"- Average Sentiment Score: {avg_sentiment:.2f}\n"

    return report

def categorize_message(message):
    # Categorize messages
    if message.startswith('http') or message.startswith('www'):
        return 'Link'
    elif message == '<Media omitted>':
        return 'Media'
    elif len(message.split()) > 10:
        return 'Long Message'
    else:
        return 'Short Message'

# Preprocessor functions (merged from preprocessor.py)
def preprocess(data):
    # Regex pattern to identify date and time
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[apAP][mM]\s-\s)'

    # Split data based on the pattern (keep date and time separately)
    messages = re.split(pattern, data)[1:]  # Avoid the first split element if it's empty
    dates = re.findall(pattern, data)  # Find all date/time entries

    # Ensure messages and dates are of the same length
    if len(dates) != len(messages) // 2:
        raise ValueError("Mismatch between number of dates and messages.")

    # Create DataFrame
    df = pd.DataFrame({'message_date': dates, 'user_message': messages[1::2]})  # Use every second item from messages

    # Clean the date string
    df['message_date'] = df['message_date'].str.replace(' - ', '', regex=False)

    # Convert to datetime
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p', errors='coerce')

    # Rename column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if len(entry) > 1:  # If message follows "user: message" structure
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    # Add user and message columns to the DataFrame
    df['user'] = users
    df['message'] = messages

    # Drop the user_message column as it's no longer needed
    df.drop(columns=['user_message'], inplace=True)

    # Extract more information from the date
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df
