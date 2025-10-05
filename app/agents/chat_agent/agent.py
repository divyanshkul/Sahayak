import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

# Set up Claude LLM - Using Sonnet 4.5 for best performance
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    max_tokens=4096,
    max_retries=6,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a teaching assistant for grade 1 to grade 6. Answer the student's questions."),
        ("human", "{text}"),
    ]
)

chain = prompt | llm

def get_chat_response(text: str) -> str:
    """Get chat response from the teaching assistant AI"""
    try:
        response = chain.invoke({"text": text})
        return response.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"