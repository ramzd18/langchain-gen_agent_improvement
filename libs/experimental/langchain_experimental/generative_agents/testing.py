from datetime import datetime, timedelta
from typing import List
from termcolor import colored
import os
import math
import faiss
import memory 
import generative_agent
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import json
# import promptLLMmemories

os.environ["OPENAI_API_KEY"] = "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"

LLM = ChatOpenAI(max_tokens=800)  # Can be any LLM you want.
USER_NAME="Person A"
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)
def addMemories(vectorstore):
    loader = TextLoader(file_path="libs/experimental/langchain_experimental/generative_agents/backend.txt")
    print("called here ")
    document=loader.load()
    vectorstore.add_documents(document)
    return vectorstore
def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
      # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
      # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
          embeddings_model.embed_query,
          index,
          InMemoryDocstore({}),
          {},
          relevance_score_fn=relevance_score_fn,
      )
    timevectorstore= TimeWeightedVectorStoreRetriever(
          vectorstore=vectorstore, other_score_keys=["importance"], k=5  
      )
    return timevectorstore
def create_agent():
  memoryretr= create_new_memory_retriever()
  newmemretr= addMemories(memoryretr)
  ram_memory = memory.GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=memoryretr,
    social_media_memory= memoryretr,
    verbose=False,
    reflection_threshold=25,  # we will give this a relatively low number to show how reflection works
)
  ram= generative_agent.GenerativeAgent(
    name="Ram",
    age=25,
    # traits="talkative,social,emphatetic",
    status="very political active",  # You can add more persistent traits here
    memory_retriever=create_new_memory_retriever(),
    education_and_work="Ram currently attends Cornell University. He is studying Computer Science and has previously worked as a Software Engineer Intern at Verizon",
    interests="hockey,football,poker,comedy,technology,",
    llm=LLM,
    memory=ram_memory,
)
  return ram; 

ram=create_agent()
# ram.memory.add_memory("Ram plays basketball with his friends and gets hurt their playing")
# ram.memory.add_memory("Ram plays soccer and slides tackle his friend")
# ram.memory.add_memory("Ram ices his limbs after playing the game to recover")
ram.memory.add_socialmedia_memory("The Minnesota Vikings are terrible they are never going to win a game")
ram.memory.add_socialmedia_memory("Where do I buy tickets for the nearest Vikings game")
ram.memory.add_socialmedia_memory("Where can I buy football tickets")
ram.memory.add_socialmedia_memory("What are some interesting things about football")
ram.memory.add_socialmedia_memory("I think Justin Jefferson is the best player in football")
ram.memory.add_socialmedia_memory("Im training everyday to play football again")
print(ram.generic_social_media_addmemories("Day in the life activities", "Use the above information to create a realistic day in the life that is very detailed of what Ram does everyday. Split activities with ;"))

# print(ram.summarize_related_memories("sports"))
print(str(ram.memory.personalitylist))
# print(ram.get_summary())


# print(ram.summarize_related_memories("sports"))
