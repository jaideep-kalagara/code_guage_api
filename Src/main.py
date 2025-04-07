import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import dotenv
import tiktoken
from openai import OpenAI

# --------------------------
# Utility Functions
# --------------------------

def count_tokens(text: str, model: str = "o3-mini") -> int:
    """
    Count the number of tokens in a given text string based on the tokenizer for a specific model.
    
    Args:
        text (str): The input text (code snippet).
        model (str): The model name to use for tokenization.
    
    Returns:
        int: Number of tokens in the text.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# --------------------------
# Setup
# --------------------------

dotenv.load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key in .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="CodeGauge API",
    description="An API that evaluates code quality and returns a rating from AI to SENIOR DEV.",
    version="1.0.0"
)

# --------------------------
# IP Restriction
# --------------------------

# List of allowed IP addresses (for now only 127.0.0.1)
# ALLOWED_IPS = ["127.0.0.1"]

# @app.middleware("http")
# async def ip_whitelist_middleware(request: Request, call_next):
#     """
#     Middleware to allow access only from specified IP addresses.

#     Raises:
#         HTTPException: 403 Forbidden if IP is not allowed.
#     """
#     client_host = request.client.host
#     if client_host not in ALLOWED_IPS:
#         raise HTTPException(status_code=403, detail=f"Access forbidden: IP {client_host} not allowed.")
#     response = await call_next(request)
#     return response

# --------------------------
# Request and Response Models
# --------------------------

class CodeReviewRequest(BaseModel):
    code: str

class CodeReviewResponse(BaseModel):
    code: str
    reason: str

# --------------------------
# Constants
# --------------------------

TOKEN_LIMIT = 1000
MODEL_NAME = "o3-mini"
SYSTEM_PROMPT = (
    "You are a senior software developer specializing in code quality, security, and best practices.\n"
    "Your task is to review code snippets submitted by junior developers.\n\n"
    "You must choose ONLY one label from this list:\n"
    "[\"AI\", \"BAD\", \"UNSAFE\", \"SPAGHETTI\", \"INCOMPLETE\", \"OK\", \"GOOD\", \"SAFE\", \"GREAT\", \"SENIOR DEV\"]\n\n"
    "Instructions:\n"
    "- Start your reply with the label only (for example: GOOD) followed by a colon.\n"
    "- After the label, write a very short reason (one or two sentences) explaining why you chose it.\n"
    "- Be strict, honest, and professional."
)

# --------------------------
# API Endpoints
# --------------------------

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint that returns a welcome HTML message.
    
    Returns:
        str: HTML welcome message.
    """
    return "<h1>Welcome to the CodeGauge API!</h1>"

@app.post("/code-review/", response_model=CodeReviewResponse, tags=["Code Review"])
async def review_code(request: CodeReviewRequest):
    """
    Analyze a submitted code snippet and return a quality rating.

    Request Body:
    - code (str): The code snippet to be reviewed.

    Response Body:
    - code (str): The assigned label (e.g., GOOD, BAD, etc.).
    - reason (str): A short explanation for the assigned label.
    """
    code = request.code.strip()
    
    # Validate code token length
    if count_tokens(code, model=MODEL_NAME) > TOKEN_LIMIT:
        raise HTTPException(
            status_code=400,
            detail="Code snippet exceeds the maximum token limit."
        )

    # Generate a completion from OpenAI
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the code submitted by the developer:\n\n{code}"}
        ],
    )

    # Split label and reason properly
    try:
        response_text = completion.choices[0].message.content.strip()
        label, reason = map(str.strip, response_text.split(":", 1))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model returned an unexpected response format."
        )

    return CodeReviewResponse(code=label, reason=reason)
