import re
import pandas as pd

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
