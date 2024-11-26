import streamlit as st
import preprocessor, helper
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


# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Page configuration with custom theme
st.set_page_config(
   page_title="WhatsApp Chat Analytics Pro",
   layout="wide",
   initial_sidebar_state="expanded",
   menu_items={
       'Get Help': 'https://github.com/yourusername/whatsapp-analyzer',
       'Report a bug': "https://github.com/yourusername/whatsapp-analyzer/issues",
       'About': "# WhatsApp Chat Analytics Pro\nVersion 2.0"
   }
)


# Custom CSS for better styling
st.markdown("""
   <style>
   .main {
       background-color: #f8f9fa;
   }
   .stButton>button {
       background-color: #4CAF50;
       color: white;
       border-radius: 5px;
   }
   .st-emotion-cache-metric {
       background-color: white;
       padding: 15px;
       border-radius: 10px;
       box-shadow: 0 2px 4px rgba(0,0,0,0.1);
   }
   h1, h2, h3 {
       color: #2c3e50;
   }
   </style>
   """, unsafe_allow_html=True)


# Main title with emoji and subtitle
st.title("📱 WhatsApp Chat Analytics Pro")
st.markdown("*Advanced Analytics and Insights for Your Conversations*")


# Sidebar with improved styling
with st.sidebar:
   st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/100px-WhatsApp.svg.png", width=100)
   st.title("Configuration")


   uploaded_file = st.file_uploader("📂 Upload Chat File", type=["txt"])


   if uploaded_file:
       bytes_data = uploaded_file.getvalue()
       data = bytes_data.decode("utf-8")
       df = preprocessor.preprocess(data)


       # User selection
       user_list = df['user'].unique().tolist()
       if 'group_notification' in user_list:
           user_list.remove('group_notification')
       user_list.sort()
       user_list.insert(0, "Overall")
       selected_user = st.selectbox("👤 Select User", user_list)


       # Time range filter
       st.subheader("📅 Time Range")
       min_date = df['date'].min().date()
       max_date = df['date'].max().date()
       start_date = st.date_input("Start Date", min_date)
       end_date = st.date_input("End Date", max_date)


       # Analysis options
       st.subheader("🔍 Analysis Options")
       show_sentiment = st.checkbox("Sentiment Analysis", True)
       show_network = st.checkbox("User Network Analysis", True)
       show_topic = st.checkbox("Topic Modeling", True)


       analyze_button = st.button("🚀 Generate Analysis")


if uploaded_file is not None and analyze_button:
   # Filter data based on date range
   mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
   df_filtered = df.loc[mask]


   # Basic Statistics in a modern card layout
   st.header("📊 Key Metrics")
   col1, col2, col3, col4 = st.columns(4)


   stats = helper.fetch_stats(selected_user, df_filtered)
   metrics = {
       "Total Messages": stats[0],
       "Total Words": stats[1],
       "Media Shared": stats[2],
       "Links Shared": stats[3]
   }


   for col, (metric, value) in zip([col1, col2, col3, col4], metrics.items()):
       col.metric(
           label=metric,
           value=f"{value:,}",
           delta=f"{(value / sum(metrics.values()) * 100):.1f}%"
       )


   # New Feature 1: Message Length Distribution
   st.subheader("📏 Message Length Distribution")
   df_filtered['message_length'] = df_filtered['message'].str.len()
   fig = px.histogram(
       df_filtered,
       x='message_length',
       nbins=50,
       title="Distribution of Message Lengths",
       labels={'message_length': 'Message Length (characters)'}
   )
   st.plotly_chart(fig, use_container_width=True)


   # New Feature 2: Response Time Analysis
   import plotly.express as px

   st.subheader("⏱️ Response Time Analysis")

   # Calculate response times in minutes
   df_filtered['response_time'] = df_filtered['date'].diff().dt.total_seconds() / 60

   # Filter extreme outliers: remove response times beyond the 99th percentile
   threshold = df_filtered['response_time'].quantile(0.99)
   df_filtered_filtered = df_filtered[df_filtered['response_time'] <= threshold]

   # Create a box plot with the filtered data
   fig = px.box(
       df_filtered_filtered,
       y='response_time',
       title="Message Response Times (minutes)",
       labels={'response_time': 'Response Time (minutes)'},
       points="all"  # Include points for outliers within the filtered data
   )

   # Update layout to enhance readability
   fig.update_layout(
       yaxis=dict(title="Response Time (minutes)", gridcolor="lightgrey"),
       xaxis=dict(showticklabels=False),
       title=dict(x=0.5),  # Center the title
       plot_bgcolor="white"
   )

   # Show plot in Streamlit
   st.plotly_chart(fig, use_container_width=True)

   # New Feature 3: Sentiment Analysis
   if show_sentiment:
       st.subheader("😊 Sentiment Analysis")
       sid = SentimentIntensityAnalyzer()
       df_filtered['sentiment'] = df_filtered['message'].apply(
           lambda x: sid.polarity_scores(x)['compound']
       )


       fig = go.Figure()
       fig.add_trace(go.Violin(
           y=df_filtered['sentiment'],
           box_visible=True,
           line_color='#4CAF50',
           fillcolor='#81C784',
           opacity=0.6
       ))
       fig.update_layout(
           title="Message Sentiment Distribution",
           yaxis_title="Sentiment Score"
       )
       st.plotly_chart(fig, use_container_width=True)


   # New Feature 4: Interactive Timeline
   st.subheader("📈 Interactive Message Timeline")
   timeline_data = helper.daily_timeline(selected_user, df_filtered)
   fig = px.line(
       timeline_data,
       x='only_date',
       y='message',
       title="Daily Message Frequency",
       line_shape="spline"
   )
   st.plotly_chart(fig, use_container_width=True)


   # New Feature 5: User Activity Patterns
   st.subheader("🕒 Hourly Activity Patterns")
   hourly_activity = df_filtered['hour'].value_counts().sort_index()
   fig = px.line_polar(
       r=hourly_activity.values,
       theta=hourly_activity.index,
       line_close=True,
       title="24-Hour Activity Pattern"
   )
   st.plotly_chart(fig, use_container_width=True)


   # New Feature 6: Word Usage Evolution
   st.subheader("📚 Word Usage Evolution")
   words_per_day = df_filtered.groupby('date')['word_count'].mean()
   fig = px.line(
       x=words_per_day.index,
       y=words_per_day.values,
       title="Average Words per Message Over Time",
       labels={'x': 'Date', 'y': 'Average Words'}
   )
   st.plotly_chart(fig, use_container_width=True)


   # New Feature 7: User Interaction Network
   if show_network and selected_user == "Overall":
       st.subheader("🔄 User Interaction Network")
       G = nx.Graph()
       user_pairs = list(zip(df_filtered['user'][:-1], df_filtered['user'][1:]))
       edge_weights = Counter(user_pairs)


       for (u1, u2), weight in edge_weights.items():
           if u1 != u2 and u1 != 'group_notification' and u2 != 'group_notification':
               G.add_edge(u1, u2, weight=weight)


       pos = nx.spring_layout(G)


       edge_x = []
       edge_y = []
       for edge in G.edges():
           x0, y0 = pos[edge[0]]
           x1, y1 = pos[edge[1]]
           edge_x.extend([x0, x1, None])
           edge_y.extend([y0, y1, None])


       fig = go.Figure()
       fig.add_trace(go.Scatter(
           x=edge_x, y=edge_y,
           line=dict(width=0.5, color='#888'),
           hoverinfo='none',
           mode='lines'
       ))


       node_x = []
       node_y = []
       for node in G.nodes():
           x, y = pos[node]
           node_x.append(x)
           node_y.append(y)


       fig.add_trace(go.Scatter(
           x=node_x, y=node_y,
           mode='markers+text',
           text=list(G.nodes()),
           textposition='top center',
           marker=dict(size=20, color='#1f77b4'),
           hoverinfo='text'
       ))


       fig.update_layout(
           title="User Interaction Network",
           showlegend=False,
           hovermode='closest'
       )
       st.plotly_chart(fig, use_container_width=True)


   # New Feature 8: Advanced Emoji Analysis
   st.subheader("🎭 Advanced Emoji Analysis")
   emoji_stats = helper.emoji_helper(selected_user, df_filtered)
   if not emoji_stats.empty:
       fig = px.treemap(
           emoji_stats,
           path=[0],
           values=1,
           title="Emoji Usage Distribution"
       )
       st.plotly_chart(fig, use_container_width=True)


   # New Feature 9: Message Type Analysis
   st.subheader("📊 Message Type Analysis")
   df_filtered['msg_type'] = df_filtered['message'].apply(helper.categorize_message)
   msg_types = df_filtered['msg_type'].value_counts()


   fig = px.pie(
       values=msg_types.values,
       names=msg_types.index,
       title="Distribution of Message Types",
       hole=0.4
   )
   st.plotly_chart(fig, use_container_width=True)


   # New Feature 10: User Engagement Score
   if selected_user != "Overall":
       st.subheader("🎯 User Engagement Analysis")
       engagement_metrics = {
           'Messages per Day': len(df_filtered) / len(df_filtered['only_date'].unique()),
           'Average Message Length': df_filtered['message'].str.len().mean(),
           'Media Share Rate': len(df_filtered[df_filtered['message'] == '<Media omitted>']) / len(df_filtered),
           'Response Rate': len(df_filtered[df_filtered['response_time'] < 60]) / len(df_filtered),
       }


       fig = px.bar(
           x=list(engagement_metrics.keys()),
           y=list(engagement_metrics.values()),
           title="User Engagement Metrics"
       )
       st.plotly_chart(fig, use_container_width=True)


   # Export options
   st.subheader("📥 Export Analysis")
   col1, col2 = st.columns(2)
   with col1:
       if st.button("Export to CSV"):
           st.download_button(
               label="Download Data",
               data=df_filtered.to_csv(index=False),
               file_name="chat_analysis.csv",
               mime="text/csv"
           )
   with col2:
       if st.button("Generate Report"):
           # Create a summary report
           report = helper.generate_report(selected_user, df_filtered)
           st.download_button(
               label="Download Report",
               data=report,
               file_name="analysis_report.md",
               mime="text/markdown"
           )


else:
   import streamlit as st


   # Welcome message and instructions with black text
   st.markdown(
       """
       <div style="color: black;">
           👋 <b>Welcome to WhatsApp Chat Analytics Pro!</b>


           <p>To get started:</p>
           <ol>
               <li>Export your WhatsApp chat (without media)</li>
               <li>Upload the exported text file</li>
               <li>Select a user or analyze overall chat</li>
               <li>Click 'Generate Analysis' to see insights</li>
           </ol>


           <p>Need help? Check out our <a href="#" style="color: blue;">documentation</a> or
           <a href="#" style="color: blue;">tutorial video</a>.</p>
       </div>
       """,
       unsafe_allow_html=True,
   )


   # Footer with black text
   st.markdown(
       """
       <hr style="border-top: 1px solid #ccc;">
       <div style="text-align: center; color: black;">
           Made with ❤️ by Your Name |
           <a href="https://github.com/3amdevvv" style="color: blue;">GitHub</a> |
           <a href="#" style="color: blue;">Documentation</a>
       </div>
       """,
       unsafe_allow_html=True,
   )
