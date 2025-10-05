import os
import asyncio
import vertexai
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_firestore import FirestoreVectorStore
from google.cloud import firestore
from google.oauth2 import service_account
import warnings

# Import SQL functionality
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
# from google.cloud.sql.connector import Connector  # Not needed for local PostgreSQL
import asyncpg
import sqlalchemy
from dotenv import load_dotenv
# from google.oauth2 import service_account  # Not needed for local PostgreSQL
from collections import defaultdict

warnings

# Load environment variables
load_dotenv()

PROJECT_ID = "krishi-saarthi-main"
OLD_PROJECT_ID = "sahayak-ai-agentic-ai-day"
LOCATION = "us-central1"

# New credentials for Vertex AI
new_credentials_path = "/Users/divyansh/Desktop/Divyansh/Development/Hackathons/AgenticAIDays/final/pragati-backend/secrets/anthropic-sahayak-main-6808fd9290d7.json"
# Old credentials for Firestore (has existing indexes)
old_credentials_path = "/Users/divyansh/Desktop/Divyansh/Development/Hackathons/AgenticAIDays/final/pragati-backend/secrets/older/sahayak-ai-credentials.json"

# Set the environment variable for Vertex AI (new credentials)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = new_credentials_path

# Initialize Vertex AI with new project
vertexai.init(project=PROJECT_ID, location=LOCATION)

# SQL Database setup - Local PostgreSQL
# Local PostgreSQL connection (no Cloud SQL connector needed)
database_url = "postgresql+asyncpg://postgres:AquaRegia@localhost:5432/multigradeschool"

# Create async engine for SQL
sql_engine = create_async_engine(
    database_url,
    echo=False,
)

# Set up the LLM - Claude Sonnet 4.5
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    max_tokens=8096,
    max_retries=6,
)

# Set up embeddings model
embeddings_model = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=PROJECT_ID,
    location=LOCATION,
)

# Create Firestore client with OLD credentials (for existing indexes)
firestore_credentials = service_account.Credentials.from_service_account_file(
    old_credentials_path
)
client = firestore.Client(
    project=OLD_PROJECT_ID,
    credentials=firestore_credentials
)

collection_name = "ncert_rag_firestore_final_test"

# Load your existing vector store
vectorstore = FirestoreVectorStore(
    collection=collection_name,
    embedding_service=embeddings_model,
    client=client
)
retriever = vectorstore.as_retriever()

# SQL function to get questions by vector_id
async def get_questions_by_vector_id(vector_id, limit=3):
    """Fetch top 3 questions from SQL database using vector_id"""
    query = text(f"""
        SELECT
            q.question,
            q.chapter_id,
            q.topic_id,
            c.vector_id,
            t.name as topic_name
        FROM questions q
        INNER JOIN chapters c ON q.chapter_id = c.chapter_id
        INNER JOIN topics t ON q.topic_id = t.topic_id
        WHERE c.vector_id = '{vector_id}'
        ORDER BY q.topic_id, q.question_id
        LIMIT {limit};
    """)

    print(f"\n{'~'*80}")
    print(f"💾 PostgreSQL Query for vector_id: '{vector_id}'")
    print(f"   Database: localhost:5432/multigradeschool")
    print(f"   Limit: {limit}")

    try:
        async with sql_engine.begin() as conn:
            result = await conn.execute(query)
            questions = result.fetchall()
            print(f"   ✅ Found {len(questions)} questions")

            if questions:
                print(f"\n   📋 Question Details:")
                for i, (question, chapter_id, topic_id, vec_id, topic_name) in enumerate(questions, 1):
                    print(f"      {i}. Chapter ID: {chapter_id}, Topic ID: {topic_id}")
                    print(f"         Topic: {topic_name}")
                    print(f"         Question: {question[:100]}{'...' if len(question) > 100 else ''}")
                    print()
            else:
                print(f"   ⚠️  No questions found for this vector_id")

            print(f"{'~'*80}\n")
            return questions
    except Exception as e:
        print(f"   ❌ SQL Error: {e}")
        print(f"{'~'*80}\n")
        return []

# Enhanced format_docs function that extracts display_name from metadata
def format_docs_with_metadata(docs):
    """Format documents and extract display_name for SQL queries"""
    formatted_content = []
    display_names = []

    print(f"\n{'='*80}")
    print(f"📄 FIRESTORE RETRIEVAL - Processing {len(docs)} documents from Firestore")
    print(f"{'='*80}")

    for i, doc in enumerate(docs, 1):
        print(f"\n📖 Document {i}:")
        print(f"   Content preview: {doc.page_content[:150]}...")
        print(f"   Metadata keys: {list(doc.metadata.keys())}")

        formatted_content.append(doc.page_content)

        # Extract display_name from nested metadata structure
        if hasattr(doc, 'metadata') and 'metadata' in doc.metadata and 'display_name' in doc.metadata['metadata']:
            display_name = doc.metadata['metadata']['display_name']
            display_names.append(display_name)
            print(f"   ✅ Found display_name: {display_name}")
        else:
            print(f"   ⚠️  No display_name found in metadata")

    print(f"\n🎯 Unique display_names extracted: {list(set(display_names))}")
    print(f"{'='*80}\n")

    return {
        'content': "\n\n".join(formatted_content),
        'display_names': list(set(display_names))  # Remove duplicates
    }

# Enhanced async function to get context with questions
async def get_context_with_questions(query):
    """Retrieve documents and get related questions for in-context learning"""
    print(f"\n🔍 QUERY: {query}")
    print(f"🗄️  Using Firestore project: {OLD_PROJECT_ID}")
    print(f"📚 Collection: {collection_name}")

    # Get retrieved documents
    docs = retriever.invoke(query)
    doc_info = format_docs_with_metadata(docs)

    # Get related questions from SQL using display_names as vector_ids
    all_questions = []
    print(f"\n💾 Fetching questions from PostgreSQL for {len(doc_info['display_names'])} display_names...")
    for display_name in doc_info['display_names']:
        questions = await get_questions_by_vector_id(display_name, limit=10)
        all_questions.extend(questions)

    print(f"✅ Total questions fetched from SQL: {len(all_questions)}")

    # Format questions for prompt
    formatted_questions = ""
    if all_questions:
        formatted_questions = "\n\nRelated Example Questions:\n"
        for i, (question, chapter_id, topic_id, vector_id, topic_name) in enumerate(all_questions[:3], 1):
            formatted_questions += f"{i}. Topic: {topic_name}\n   Question: {question}\n\n"

    return {
        'context': doc_info['content'],
        'questions': formatted_questions,
        'display_names': doc_info['display_names']
    }

# Create the enhanced prompt template with in-context learning
prompt = ChatPromptTemplate.from_template("""
Generate 5 questions based on the following context from NCERT textbooks:

{context}
{questions}

Request: {question}

Instructions: 
- Generate exactly 5 questions based on the provided context
- Use the example questions above as reference for the type and style of questions expected
- Each question should focus on a different topic/concept from the context
- Divide the topics equally among the 5 questions
- Return the response in the following JSON format only (no additional text):

{{
  "question_1": {{
    "topic": "Topic Name 1",
    "question": "Your first question here?"
  }},
  "question_2": {{
    "topic": "Topic Name 2", 
    "question": "Your second question here?"
  }},
  "question_3": {{
    "topic": "Topic Name 3",
    "question": "Your third question here?"
  }},
  "question_4": {{
    "topic": "Topic Name 4",
    "question": "Your fourth question here?"
  }},
  "question_5": {{
    "topic": "Topic Name 5",
    "question": "Your fifth question here?"
  }}
}}

Generate the JSON response:
""")

# Your post-processing function (kept for backward compatibility)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Enhanced async RAG chain with in-context learning
async def enhanced_rag_chain(question):
    """Enhanced RAG chain that includes in-context learning with SQL questions"""
    try:
        # Get context and related questions
        context_data = await get_context_with_questions(question)
        
        print(f"🔍 Retrieved from: {context_data['display_names']}")
        question_count = len(context_data['questions'].split('Question:')) - 1 if context_data['questions'] else 0
        print(f"📋 Found {question_count} related questions")
        
        # Create the prompt with enhanced context
        formatted_prompt = prompt.format(
            context=context_data['context'],
            questions=context_data['questions'],
            question=question
        )
        
        # Get response from LLM
        response = await llm.ainvoke(formatted_prompt)
        return response.content
        
    except Exception as e:
        print(f"❌ Error in enhanced RAG chain: {e}")
        # Fallback to basic RAG if enhanced version fails
        basic_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt.partial(questions="")  # Empty questions for fallback
            | llm
            | StrOutputParser()
        )
        return basic_chain.invoke(question)

# Original RAG chain (for backward compatibility)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt.partial(questions="")  # Empty questions for basic usage
    | llm
    | StrOutputParser()
)

# Renamed main function for API usage
async def invoke_shikshak_agent(question: str):
    """Invoke Shikshak Mitra agent with enhanced RAG and in-context learning"""
    import json
    try:
        print(f"\n{'*'*80}")
        print(f"🤔 REQUEST: {question}")
        print(f"{'*'*80}")

        # Use enhanced RAG chain
        raw_answer = await enhanced_rag_chain(question)
        print(f"\n{'='*80}")
        print(f"📝 RAW MODEL RESPONSE ({len(raw_answer)} chars):")
        print(f"{'='*80}")
        print(raw_answer)
        print(f"{'='*80}\n")

        # Try to parse the JSON response
        try:
            # Clean the response - remove any markdown formatting
            cleaned_answer = raw_answer.strip()
            if cleaned_answer.startswith('```json'):
                cleaned_answer = cleaned_answer[7:]
            if cleaned_answer.endswith('```'):
                cleaned_answer = cleaned_answer[:-3]
            cleaned_answer = cleaned_answer.strip()

            # Parse JSON
            structured_response = json.loads(cleaned_answer)

            print(f"\n{'#'*80}")
            print(f"✅ SUCCESSFULLY GENERATED {len(structured_response)} QUESTIONS:")
            print(f"{'#'*80}")

            for key, value in structured_response.items():
                if key.startswith('question_'):
                    q_num = key.split('_')[1]
                    print(f"\n📌 Question {q_num}:")
                    if isinstance(value, dict):
                        print(f"   Topic: {value.get('topic', 'N/A')}")
                        print(f"   Question: {value.get('question', 'N/A')}")
                    else:
                        print(f"   {value}")

            print(f"\n{'#'*80}")
            print(f"✅ This is the ACTUAL RESPONSE being sent to frontend")
            print(f"{'#'*80}\n")

            return structured_response

        except json.JSONDecodeError as e:
            print(f"\n{'!'*80}")
            print(f"⚠️ JSON PARSING FAILED - FALLBACK RESPONSE")
            print(f"{'!'*80}")
            print(f"Error: {e}")
            print(f"This is likely what your frontend is receiving!")
            print(f"{'!'*80}\n")
            # Fallback to raw response if JSON parsing fails
            return {"raw_response": raw_answer, "parse_error": str(e)}

    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"❌ ERROR OCCURRED: {e}")
        print(f"{'!'*80}\n")
        raise e

# Main function for standalone testing
async def main():
    """Main function for standalone testing"""
    try:
        question = "Generate questions related to angles"
        answer = await invoke_shikshak_agent(question)
        return answer
    finally:
        # Clean up connections
        await sql_engine.dispose()
        print("\n🔒 Connections closed successfully")

# Use it
if __name__ == "__main__":
    asyncio.run(main())