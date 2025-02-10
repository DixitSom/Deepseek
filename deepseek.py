import os
import json
import re
from datetime import datetime

import redis
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI, HTTPException

# from transformers import LlamaTokenizer, LlamaForCausalLM
# import torch

load_dotenv()

app = FastAPI()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # If using cloud API
GPT_API_KEY = os.getenv("CHAT_GPT_API_KEY")  # If using cloud API
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
BASE_URL = "https://api.deepseek.com"
# LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Change to a smaller model if needed
# LLAMA_MODEL = "meta-llama/Llama-2-7b-chat"

# Initialize Redis
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

# Load Model and Tokenizer (for local inference)
# tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
# model = LlamaForCausalLM.from_pretrained(
#     LLAMA_MODEL, torch_dtype=torch.float16, device_map="auto"
# )

# In-memory storage for database connections
db_connections = {}
# client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com")
client = OpenAI(api_key=GPT_API_KEY)


class DBConnect(BaseModel):
    db_url: str
    username: str
    password: str


class UserQuery(BaseModel):
    question: str
    db_url: str


@app.post("/connect_db")
async def connect_db(credentials: DBConnect):
    try:
        engine = create_engine(credentials.db_url)
        # conn = engine.connect()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        conn: Session = SessionLocal()
        db_connections[credentials.db_url] = engine

        # Extract schema
        schema_query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """
        result = conn.execute(text(schema_query)).fetchall()

        schema_dict = {}
        for table, column, dtype in result:
            if table not in schema_dict:
                schema_dict[table] = []
            schema_dict[table].append({"column": column, "type": dtype})

        # Store schema in Redis
        redis_client.set(f"schema:{credentials.db_url}", json.dumps(schema_dict))

        conn.close()

        return {"message": "Database connected and schema stored"}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Database connection failed: {str(e)}"
        )


def generate_sql_query(prompt: str):
    """Generate a response using Llama 2"""
    # tokenizer.pad_token = tokenizer.eos_token
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # inputs = {key: val for key, val in inputs.items()}  # Move to GPU if available

    # with torch.no_grad():
    #     output = model.generate(**inputs, max_new_tokens=512)

    # return tokenizer.decode(output[0], skip_special_tokens=True)
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are a SQL and data analysis expert."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     stream=False,
    # )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a SQL and data analysis expert. Only provide SQL query in answer if asked for SQL",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
        store=True,
    )
    return response.choices[0].message.content


def generate_response(prompt: str, sql_result: str, sql_query: str):
    """Generate a response using Llama 2"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"You are a technical asisstant. Respond to the question with data as follows {sql_result} for sql query that system run for user is {sql_query}",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
        store=True,
    )
    return response.choices[0].message.content


@app.post("/ask")
def ask_question(query: UserQuery):
    if query.db_url not in db_connections:
        raise HTTPException(status_code=400, detail="Database not connected")

    try:
        # Retrieve schema from Redis
        schema_data = redis_client.get(f"schema:{query.db_url}")
        if schema_data:
            schema_text = json.loads(schema_data)
            schema_context = "\n".join(
                [
                    f"Table {t}: "
                    + ", ".join([f"{c['column']} ({c['type']})" for c in v])
                    for t, v in schema_text.items()
                ]
            )
        else:
            schema_context = "No schema available"

        # Generate SQL with schema awareness
        prompt = f"""
        Database Schema:
        {schema_context}
        
        Convert this question into an SQL query: '{query.question}'
        """
        print(prompt)
        sql_query = generate_sql_query(prompt)
        print("query", sql_query)
        match = re.search(r"```sql\n(.*?)\n```", sql_query, re.DOTALL)
        sql_query = match.group(1) if match else None

        # Execute SQL query
        engine = db_connections[query.db_url]
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        conn: Session = SessionLocal()
        # with engine.connect() as connection:
        result = conn.execute(text(sql_query)).fetchall()
        result_data = [tuple(row) for row in result]
        conn.close()

        for row in result_data:
            list_row = list(row)
            for item in list_row:
                if isinstance(item, datetime):
                    list_row[list_row.index(item)] = item.strftime("%Y-%m-%d %H:%M:%S")

            result_data[result_data.index(row)] = tuple(list_row)

        # Generate natural response
        result_text = json.dumps(result_data)
        summary = generate_response(query.question, result_text, sql_query)

        return {"sql_query": sql_query, "data": result_data, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
