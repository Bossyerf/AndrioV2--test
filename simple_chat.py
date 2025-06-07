#!/usr/bin/env python3
"""Simple asynchronous chat loop using Ollama.
Stores conversation logs in ChromaDB for future learning."""

import asyncio
from datetime import datetime
import time
import logging
import uuid

import aiohttp
import chromadb
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleChat:
    """Maintain chat history and store conversation logs."""

    def __init__(self, model: str = "andriov2-qwen3",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = model
        self.history = []  # Ollama chat format
        self.url = "http://localhost:11434/api/chat"
        self.conversation_id = str(uuid.uuid4())

        self.embedding_model = SentenceTransformer(embedding_model)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda")

        self.chroma = chromadb.PersistentClient(path="./andrio_knowledge_db")
        self.collection = self.chroma.get_or_create_collection(name="chat_logs")

    async def send(self, message: str) -> str:
        """Send a message and return the assistant response."""
        self.history.append({"role": "user", "content": message})
        async with aiohttp.ClientSession() as session:
            payload = {"model": self.model, "messages": self.history}
            async with session.post(self.url, json=payload) as resp:
                if resp.status != 200:
                    logger.error("LLM query failed: HTTP %s", resp.status)
                    response_text = f"Error: HTTP {resp.status}"
                else:
                    data = await resp.json()
                    response_text = data.get("message", {}).get("content", "")
        self.history.append({"role": "assistant", "content": response_text})
        await self._store_pair(message, response_text)
        return response_text

    async def _store_pair(self, user_msg: str, assistant_msg: str) -> None:
        """Store user and assistant messages in ChromaDB."""
        try:
            docs = [user_msg, assistant_msg]
            embeds = self.embedding_model.encode(docs)
            timestamp = datetime.now().isoformat()
            ids = [
                f"{self.conversation_id}_{int(time.time()*1000)}_user",
                f"{self.conversation_id}_{int(time.time()*1000)}_assistant",
            ]
            metadatas = [
                {"role": "user", "timestamp": timestamp,
                 "conversation": self.conversation_id},
                {"role": "assistant", "timestamp": timestamp,
                 "conversation": self.conversation_id},
            ]
            self.collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
                embeddings=[embeds[0].tolist(), embeds[1].tolist()],
            )
        except Exception as e:
            logger.error("Failed to store chat logs: %s", e)


async def chat_main() -> None:
    """Entry point for interactive chat."""
    chat = SimpleChat()
    print("💬 Simple AndrioV2 Chat")
    print("Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            response = await chat.send(user_input)
            print(f"Andrio: {response}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(chat_main())
