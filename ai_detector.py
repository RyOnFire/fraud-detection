from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def detect_columns(df):
    prompt = f"""
    I have a financial transaction CSV with these columns: {list(df.columns)}
    
    Here are the first 3 rows:
    {df.head(3).to_string()}
    
    Identify which column is most likely:
    1. The transaction amount
    2. The date or time
    3. The fraud label (0/1 or True/False) if it exists, otherwise say None
    
    Respond ONLY with valid JSON like this:
    {{"amount_col": "column_name", "time_col": "column_name", "target_col": "column_name or None"}}
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.choices[0].message.content.strip()
    text = text.replace('```json', '').replace('```', '').strip()
    
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Fallback to first columns if AI fails
        result = {
            'amount_col': df.columns[0],
            'time_col': df.columns[0],
            'target_col': 'None'
        }
    
    return result