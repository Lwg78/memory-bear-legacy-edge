import os
import time
import math
import logging
import sys

# --- CONFIGURATION FOR MACBOOK AIR 2017 (Intel Chip) ---
# We use Python's standard logging to keep things clean
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("MemoryBear")

# Hardware check warning
try:
    import numpy as np
    # Check for the specific version conflict you faced earlier
    if int(np.__version__.split('.')[0]) >= 2:
        print("\n⚠️  CRITICAL WARNING: You have NumPy 2.x installed.")
        print("This will crash on your Mac. Please run: pip install \"numpy<2.0\" \n")
        time.sleep(2)
except ImportError:
    pass

# Try importing the heavy AI libraries with error handling
try:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    import chromadb
    from sentence_transformers import SentenceTransformer
    import networkx as nx
except ImportError as e:
    print(f"\n❌ Missing libraries! Error: {e}")
    print("Please run this command in your terminal:")
    print("pip install llama-cpp-python chromadb networkx sentence-transformers huggingface_hub")
    sys.exit()

# --- CONSTANTS ---
# We use Phi-3 Mini because it is small (2.4GB) but very smart.
# It fits in the 8GB RAM of a 2017 MacBook Air.
MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
MODEL_FILE = "Phi-3-mini-4k-instruct-q4.gguf"
MODEL_PATH = f"./models/{MODEL_FILE}"

# --- COMPONENT 1: THE COGNITIVE STORAGE (Hippocampus) ---
class CognitiveStorage:
    def __init__(self):
        print("\n[1/3] Initializing Hippocampus (Memory Systems)...")
        
        # 1. Episodic Memory (Vector DB) - Stores "Experiences"
        # We use a local folder './memory_db' so memories persist after you restart
        self.chroma_client = chromadb.PersistentClient(path="./memory_db")
        self.episodic_collection = self.chroma_client.get_or_create_collection("episodic_memory")
        
        # 2. Semantic Memory (Knowledge Graph) - Stores "Facts"
        self.knowledge_graph = nx.DiGraph()
        
        # 3. Embedding Model - Converts text to numbers
        # all-MiniLM-L6-v2 is very fast and works well on Intel CPUs
        logger.info("Loading Embedding Model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_episodic_memory(self, content, strength=1.0):
        """
        Stores an event (chat log) with a timestamp and initial strength.
        """
        vector = self.encoder.encode(content).tolist()
        timestamp = time.time()
        # Create a unique ID for the memory
        mem_id = f"mem_{int(timestamp)}_{abs(hash(content))}"
        
        self.episodic_collection.add(
            documents=[content],
            embeddings=[vector],
            metadatas=[{"timestamp": timestamp, "strength": strength, "type": "episodic"}],
            ids=[mem_id]
        )
        # Only log if it's not the initial load
        logger.info(f"💾 Saved Episode: {content[:40]}...")

    def add_semantic_fact(self, subject, relation, object_):
        """
        Stores a fact in the graph. E.g., (User, loves, AI).
        """
        self.knowledge_graph.add_edge(subject, object_, relation=relation)
        logger.info(f"🕸️  Linked Fact: {subject} --{relation}--> {object_}")

    def retrieve_memories(self, query_text):
        """
        THE MEMORY BEAR ALGORITHM:
        Retrieves memories but filters them using the Ebbinghaus Forgetting Curve.
        """
        query_vec = self.encoder.encode(query_text).tolist()
        current_time = time.time()
        
        # 1. Raw Retrieval (Vector Search)
        # We ask the database for the 5 most mathematically similar memories
        results = self.episodic_collection.query(
            query_embeddings=[query_vec],
            n_results=5 
        )
        
        valid_memories = []
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                mem_id = results['ids'][0][i]
                
                # 2. Ebbinghaus Decay Calculation
                # Formula: R = e^(-t/S)
                # t = time passed, S = strength of memory
                # We scale time to 'hours' for this demo so you can see effects quickly.
                time_diff_hours = (current_time - meta['timestamp']) / 3600
                strength = meta['strength']
                
                retention = math.exp(-time_diff_hours / strength)
                
                # 3. The Cognitive Filter
                # If retention is high enough (> 0.35), we remember it.
                if retention > 0.35:
                    valid_memories.append(doc)
                    
                    # 4. Reinforcement (Spaced Repetition)
                    # Accessing the memory makes it stronger!
                    new_strength = strength + 0.5
                    self.episodic_collection.update(
                        ids=[mem_id],
                        metadatas=[{"timestamp": current_time, "strength": new_strength, "type": "episodic"}]
                    )
                    # logger.info(f"🔥 Memory Reinforced: {doc[:20]}... (New Strength: {new_strength})")
                else:
                    # logger.warning(f"👻 Memory Faded (Forgotten): {doc[:20]}...")
                    pass
                    
        return valid_memories

    def retrieve_facts(self, query_text):
        """
        Simple keyword matching to find facts in the graph.
        """
        found_facts = []
        words = query_text.lower().split()
        for node in self.knowledge_graph.nodes():
            if str(node).lower() in words:
                neighbors = self.knowledge_graph[node]
                for neighbor, attr in neighbors.items():
                    relation = attr.get('relation', 'related to')
                    found_facts.append(f"{node} {relation} {neighbor}")
        return found_facts

# --- COMPONENT 2: THE BRAIN (Transformer) ---
class LocalBrain:
    def __init__(self):
        print("[2/3] Checking for Brain Model (Phi-3)...")
        # Auto-download logic
        if not os.path.exists("./models"):
            os.makedirs("./models")
        
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading Model... (This is ~2.4GB, please wait 2-5 mins)")
            print("    Your fan might spin up. Do not close the window.")
            hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="./models")
            print("✅ Download Complete.")
        
        print("[3/3] Loading LLM into RAM...")
        # N_CTX=2048 prevents your 8GB RAM from filling up
        # n_threads=4 uses your Intel CPU efficiently
        self.llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

    def think(self, user_input, context_memories, context_facts):
        # Phi-3 Prompt Template (System -> User -> Assistant)
        mem_str = "\n".join([f"- {m}" for m in context_memories])
        fact_str = "\n".join([f"- {f}" for f in context_facts])
        
        system_msg = (
            "You are Memory Bear, a helpful AI assistant running locally."
            f"\n\nRELEVANT MEMORIES (Past Chats):\n{mem_str}"
            f"\n\nKNOWN FACTS (Context):\n{fact_str}"
            "\n\nUse these memories to answer the user briefly."
        )
        
        prompt = f"<|system|>\n{system_msg}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        
        output = self.llm(
            prompt,
            max_tokens=150, # Keep it short for speed on Intel CPU
            stop=["<|end|>"],
            echo=False
        )
        return output['choices'][0]['text'].strip()

# --- COMPONENT 3: THE AGENT (Orchestrator) ---
def main():
    print("\n" + "="*50)
    print("   🐻 MEMORY BEAR (Legacy Mac Edition) STARTED   ")
    print("="*50)
    
    # Initialize components
    storage = CognitiveStorage()
    brain = LocalBrain()
    
    # Pre-seed some semantic knowledge
    storage.add_semantic_fact("User", "is using", "MacBook Air 2017")
    storage.add_semantic_fact("Memory Bear", "runs on", "Intel CPU")
    
    print("\n✅ System Ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Saving memories... Goodbye! 👋")
                break
            
            if not user_input.strip():
                continue
                
            # 1. Retrieve
            episodic = storage.retrieve_memories(user_input)
            semantic = storage.retrieve_facts(user_input)
            
            # 2. Think & Generate
            print("Thinking... (CPU working)...", end="\r")
            response = brain.think(user_input, episodic, semantic)
            
            # Clear the "Thinking..." line
            print(" " * 30, end="\r")
            
            # 3. Respond
            print(f"Bear: {response}")
            
            # 4. Consolidate (Save Memory)
            storage.add_episodic_memory(f"User: {user_input} | Bear: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break

if __name__ == "__main__":
    main()