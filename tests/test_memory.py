
import sys
import os
import shutil
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.vector import VectorMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_memory():
    print("🧠 Testing Vector Memory...")
    
    # Use a test persistence path
    test_db_path = "data/test_memory_db"
    
    # Clean up previous test
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        
    try:
        # Initialize
        memory = VectorMemory(persistence_path=test_db_path)
        
        if not memory.enabled:
            print("❌ Vector memory disabled (dependencies missing?)")
            return
            
        print("✅ Vector memory initialized")
        
        # Test 1: Add memory
        print("1. Adding memories...")
        memory.add_memory("I feel really sad about my cat passing away.", metadata={"emotion": "sad"})
        memory.add_memory("My name is Tunda and I like to code.", metadata={"emotion": "neutral"})
        memory.add_memory("I am so happy that I got a promotion!", metadata={"emotion": "happy"})
        
        # Test 2: Retrieve relevant
        print("2. Retrieving relevant memories for 'loss'...")
        results = memory.retrieve_relevant("I am dealing with a loss", limit=2)
        
        found_cat = False
        for res in results:
            print(f"   - Found: {res['text']} (distance: {res['distance']:.4f})")
            if "cat" in res['text']:
                found_cat = True
        
        if found_cat:
            print("✅ Successfully retrieved relevant memory about cat")
        else:
            print("❌ Failed to retrieve relevant memory")
            
        # Test 3: Retrieve relevant for happiness
        print("3. Retrieving relevant memories for 'work'...")
        results = memory.retrieve_relevant("Work is going great", limit=1)
        
        found_promo = False
        for res in results:
            print(f"   - Found: {res['text']} (distance: {res['distance']:.4f})")
            if "promotion" in res['text']:
                found_promo = True
                
        if found_promo:
            print("✅ Successfully retrieved relevant memory about promotion")
        else:
            print("❌ Failed to retrieve relevant memory")

    finally:
        # Cleanup
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        print("🧹 Cleanup complete")

if __name__ == "__main__":
    test_vector_memory()
