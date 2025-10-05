import os
import json
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
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
# from google.cloud.sql.connector import Connector  # Not needed for local PostgreSQL
from dotenv import load_dotenv
# from google.oauth2 import service_account  # Not needed for local PostgreSQL
from langchain_core.messages import HumanMessage
import base64
from typing import Dict, Any


class PrabhandhakAgent:
    def __init__(self):
        warnings.filterwarnings('ignore')
        load_dotenv()

        self.PROJECT_ID = "krishi-saarthi-main"
        self.OLD_PROJECT_ID = "sahayak-ai-agentic-ai-day"
        self.LOCATION = "us-central1"

        # New credentials for Vertex AI
        self.new_credentials_path = "/Users/divyansh/Desktop/Divyansh/Development/Hackathons/AgenticAIDays/final/pragati-backend/secrets/anthropic-sahayak-main-6808fd9290d7.json"
        # Old credentials for Firestore (has existing indexes)
        self.old_credentials_path = "/Users/divyansh/Desktop/Divyansh/Development/Hackathons/AgenticAIDays/final/pragati-backend/secrets/older/sahayak-ai-credentials.json"

        # Set environment variable for Vertex AI (new credentials)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.new_credentials_path

        # Initialize Vertex AI with new project (needed for embeddings only)
        vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)

        # Initialize OCR LLM - Claude Sonnet 4.5 with vision support
        self.llm_ocr = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
            max_tokens=8096,
        )

        # Set up the main LLM - Claude Sonnet 4.5
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
            max_tokens=8096,
            max_retries=6,
        )

        # Set up embeddings model
        self.embeddings_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=self.PROJECT_ID,
            location=self.LOCATION,
        )

        # Create Firestore client with OLD credentials (for existing indexes)
        firestore_credentials = service_account.Credentials.from_service_account_file(
            self.old_credentials_path
        )
        self.client = firestore.Client(
            project=self.OLD_PROJECT_ID,
            credentials=firestore_credentials
        )
        self.collection_name = "thefastandfourier_ncert_final"
        
        # Load vector store
        self.vectorstore = FirestoreVectorStore(
            collection=self.collection_name,
            embedding_service=self.embeddings_model,
            client=self.client
        )
        self.retriever = self.vectorstore.as_retriever()
        
        # Initialize SQL connection (local PostgreSQL)
        self._init_sql_connection()
        
        # Create the enhanced prompt template
        self.prompt = ChatPromptTemplate.from_template("""
Based on the following extracted text from NCERT textbooks and related context, generate 10 educational questions with topics in JSON format:

Extracted Text: {question}

Context from NCERT: {context}

Related Example Questions: {questions}

Instructions: 
- Generate exactly 10 questions based on the extracted text and context
- Each question should have a relevant topic/subject area
- Return ONLY valid JSON in this exact format:
{{
  "Topic_1": "Topic Name 1",
  "Question_1": "Question text 1",
  "Topic_2": "Topic Name 2", 
  "Question_2": "Question text 2",
  "Topic_3": "Topic Name 3",
  "Question_3": "Question text 3",
  "Topic_4": "Topic Name 4",
  "Question_4": "Question text 4",
  "Topic_5": "Topic Name 5",
  "Question_5": "Question text 5",
  "Topic_6": "Topic Name 6",
  "Question_6": "Question text 6",
  "Topic_7": "Topic Name 7",
  "Question_7": "Question text 7",
  "Topic_8": "Topic Name 8",
  "Question_8": "Question text 8",
  "Topic_9": "Topic Name 9",
  "Question_9": "Question text 9",
  "Topic_10": "Topic Name 10",
  "Question_10": "Question text 10"
}}

- Make questions educational and appropriate for the grade level
- Cover different aspects and difficulty levels
- Use the context and example questions as reference for question style
- Return ONLY the JSON, no additional text

JSON Response:
""")

    def _init_sql_connection(self):
        """Initialize SQL connection - Local PostgreSQL"""
        # Local PostgreSQL connection (no Cloud SQL connector needed)
        database_url = "postgresql+asyncpg://postgres:AquaRegia@localhost:5432/multigradeschool"

        # Create async engine for SQL
        self.sql_engine = create_async_engine(
            database_url,
            echo=False,
        )

    async def get_questions_by_vector_id(self, vector_id, limit=3):
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
            async with self.sql_engine.begin() as conn:
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

    def format_docs_with_metadata(self, docs):
        """Format documents and extract display_name for SQL queries"""
        formatted_content = []
        display_names = []
        
        print(f"\nProcessing {len(docs)} retrieved documents:")
        print("=" * 60)
        
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content Preview: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            
            formatted_content.append(doc.page_content)
            
            # Extract display_name from nested metadata structure
            if hasattr(doc, 'metadata') and 'metadata' in doc.metadata and 'display_name' in doc.metadata['metadata']:
                display_name = doc.metadata['metadata']['display_name']
                display_names.append(display_name)
                print(f"Found display_name: {display_name}")
            else:
                print(f"No display_name found in nested metadata structure")
        
        print("=" * 60)
        print(f"Unique display_names extracted: {list(set(display_names))}")
        
        return {
            'content': "\n\n".join(formatted_content),
            'display_names': list(set(display_names))
        }

    async def get_context_with_questions(self, query):
        """Retrieve documents and get related questions for in-context learning"""
        # Get retrieved documents
        docs = self.retriever.invoke(query)
        doc_info = self.format_docs_with_metadata(docs)
        
        # Get related questions from SQL using display_names as vector_ids
        all_questions = []
        for display_name in doc_info['display_names']:
            questions = await self.get_questions_by_vector_id(display_name, limit=20)
            all_questions.extend(questions)
        
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

    def format_docs(self, docs):
        """Basic document formatting for backward compatibility"""
        return "\n\n".join(doc.page_content for doc in docs)

    async def enhanced_rag_chain(self, question):
        """Enhanced RAG chain that includes in-context learning with SQL questions"""
        try:
            # Get context and related questions
            context_data = await self.get_context_with_questions(question)
            
            print(f"Retrieved context from display_names: {context_data['display_names']}")
            print(f"Found {len(context_data['questions'].split('Question:')) - 1 if context_data['questions'] else 0} related questions")
            
            # Create the prompt with enhanced context
            formatted_prompt = self.prompt.format(
                context=context_data['context'],
                questions=context_data['questions'],
                question=question
            )
            
            # Get response from LLM
            response = await self.llm.ainvoke(formatted_prompt)
            return response.content
            
        except Exception as e:
            print(f"Error in enhanced RAG chain: {e}")
            # Fallback to basic RAG if enhanced version fails
            basic_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt.partial(questions="")
                | self.llm
                | StrOutputParser()
            )
            return basic_chain.invoke(question)

    def encode_image_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 for OCR"""
        return base64.b64encode(image_bytes).decode('utf-8')

    async def process_image_ocr(self, image_bytes: bytes) -> Dict[str, Any]:
        """Process image for OCR and answer extraction"""
        try:
            # Encode image for OCR
            base64_image = self.encode_image_bytes(image_bytes)

            # Create OCR message
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "Please extract and transcribe all the readable text from this image of a book page. "
                               "Maintain paragraphs and formatting as much as possible. Do not summarize or interpret—just transcribe the text exactly as it appears."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ]
            )

            # Extract text from image using OCR
            print("Extracting text from image...")
            ocr_result = self.llm_ocr.invoke([message])
            extracted_question = ocr_result.content
            print(f"Extracted Question:\n{extracted_question}\n")
            
            # Use enhanced RAG chain to generate questions
            print("\n{'*'*80}")
            print("🔄 Processing with Enhanced RAG...")
            print(f"{'*'*80}\n")

            questions_response = await self.enhanced_rag_chain(extracted_question)

            print(f"\n{'='*80}")
            print(f"📝 RAW MODEL RESPONSE ({len(questions_response)} chars):")
            print(f"{'='*80}")
            print(questions_response)
            print(f"{'='*80}\n")

            # Parse JSON response
            try:
                # Clean the response to extract only JSON
                json_start = questions_response.find('{')
                json_end = questions_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = questions_response[json_start:json_end]
                    questions_json = json.loads(json_str)
                else:
                    questions_json = json.loads(questions_response)

                print(f"\n{'#'*80}")
                print(f"✅ SUCCESSFULLY GENERATED {len([k for k in questions_json.keys() if k.startswith('Question_')])} QUESTIONS:")
                print(f"{'#'*80}")

                for i in range(1, 11):
                    topic_key = f"Topic_{i}"
                    question_key = f"Question_{i}"
                    if topic_key in questions_json and question_key in questions_json:
                        print(f"\n📌 Question {i}:")
                        print(f"   Topic: {questions_json[topic_key]}")
                        print(f"   Question: {questions_json[question_key]}")

                print(f"\n{'#'*80}")
                print(f"✅ This is the ACTUAL RESPONSE being sent to frontend")
                print(f"{'#'*80}\n")

            except json.JSONDecodeError as e:
                print(f"\n{'!'*80}")
                print(f"⚠️ JSON PARSING FAILED - FALLBACK RESPONSE")
                print(f"{'!'*80}")
                print(f"Error: {e}")
                print(f"Using fallback question!")
                print(f"{'!'*80}\n")
                # Fallback if JSON parsing fails
                questions_json = {
                    "Topic_1": "General Knowledge",
                    "Question_1": "Based on the extracted text, what is the main concept being discussed?"
                }
            
            return {
                "status": "success",
                "extracted_text": extracted_question,
                "questions": questions_json,
                "message": "OCR and question generation completed successfully"
            }
            
        except Exception as e:
            print(f"Error occurred: {e}")
            return {
                "status": "error",
                "message": f"Error processing image: {str(e)}",
                "extracted_text": None,
                "questions": None
            }
        finally:
            # Clean up connections
            try:
                await self.sql_engine.dispose()
                print("\nConnections closed successfully")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")