# from typing import List, Annotated

# from sqlmodel import Field, SQLModel, create_engine, Session, select
# from fastapi import FastAPI, Depends, HTTPException

# app = FastAPI()


# class ItemBase(SQLModel):
#     """Item Base Model

#     Args:
#         SQLModel (): Base Model
#     """

#     name: str
#     description: str = None
#     price: float
#     tax: float = None


# class Item(ItemBase, table=True):
#     """Item Model

#     Args:
#         SQLModel (): Base Model
#         table (bool, optional): to create table on db. Defaults to True.
#     """

#     id: int | None = Field(primary_key=True, default=None)
#     # name: str = Field(index=True)
#     # description: str = None
#     # price: float
#     # tax: float = None


# SQLITE_FILE_NAME = "database.db"
# SQLITE_URL = f"sqlite:///{SQLITE_FILE_NAME}"
# connect_args = {"check_same_thread": False}
# engine = create_engine(SQLITE_URL, connect_args=connect_args)


# def create_db_and_tables():
#     """Create database and tables"""
#     SQLModel.metadata.create_all(engine)


# def get_session():
#     """Get Session

#     Yields:
#         Session: Session object
#     """
#     with Session(engine) as session:
#         yield session


# @app.on_event("startup")
# def on_startup():
#     """On Startup Event create database and tables"""
#     create_db_and_tables()


# SessionDep = Annotated[Session, Depends(get_session)]


# @app.get("/")
# async def root():
#     """Root

#     Returns:
#         dict: Hello World
#     """
#     return {"message": "Hello World"}


# @app.post("/items/")
# async def create_item(item: ItemBase, session: SessionDep) -> Item:
#     """Create Item

#     Args:
#         item (Item): Item object
#         session (SessionDep): Session Dependency to interact with the database

#     Returns:
#         Item: Created Item
#     """
#     db_item = Item.model_validate(item)
#     session.add(db_item)
#     session.commit()
#     session.refresh(db_item)
#     return db_item


# @app.get("/items/")
# async def read_items(
#     session: SessionDep, limit: int = 100, offset: int = 0
# ) -> List[Item]:
#     """Get items

#     Args:
#         session (SessionDep): Session Dependency to interact with the database
#         limit (int, optional): Limit Items . Defaults to 100.
#         offset (int, optional): Offset Items . Defaults to 0.

#     Returns:
#         List[Item]: List of Items
#     """
#     items = session.exec(select(Item).offset(offset).limit(limit)).all()
#     return items


# @app.get("/items/{item_id}")
# async def read_item(session: SessionDep, item_id: int) -> Item | None:
#     """Get item by id

#     Args:
#         session (SessionDep): Session Dependency to interact with the database
#         item_id (int): Item id

#     Raises:
#         HTTPException: Item not found

#     Returns:
#         Item | None: Item object
#     """
#     item = session.get(Item, item_id)
#     if not item:
#         raise HTTPException(status_code=404, detail="Item not found")
#     return item

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import openai
import os
from dotenv import load_dotenv
from jose import JWTError, jwt
from datetime import datetime, timedelta
import psycopg2
import redis
import json

# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


load_dotenv()

app = FastAPI()

# Load environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# In-memory storage for simplicity (use a database in production)
db_connections = {}


class DBConnect(BaseModel):
    db_url: str
    username: str
    password: str


class UserQuery(BaseModel):
    question: str


# Token generation
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@app.post("/connect_db")
def connect_db(credentials: DBConnect):
    try:
        engine = create_engine(credentials.db_url)
        conn = engine.connect()
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

        return {"message": "Database connected and schema stored"}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Database connection failed: {str(e)}"
        )


@app.post("/ask")
def ask_question(query: UserQuery, db_url: str):
    if db_url not in db_connections:
        raise HTTPException(status_code=400, detail="Database not connected")

    try:
        # Retrieve schema from Redis
        schema_data = redis_client.get(f"schema:{db_url}")
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
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert."},
                {"role": "user", "content": prompt},
            ],
        )
        sql_query = response["choices"][0]["message"]["content"]

        # Execute SQL query
        engine = db_connections[db_url]
        with engine.connect() as connection:
            result = connection.execute(text(sql_query)).fetchall()
            result_data = [dict(row) for row in result]

        # Generate natural response
        result_text = str(result_data)
        final_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize this database result."},
                {"role": "user", "content": result_text},
            ],
        )
        summary = final_response["choices"][0]["message"]["content"]

        return {"sql_query": sql_query, "data": result_data, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Run with `uvicorn filename:app --reload`
