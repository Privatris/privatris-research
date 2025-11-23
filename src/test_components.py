import unittest
from src.agent import RLAgent
from src.data_loader import DataLoader

class TestComponents(unittest.TestCase):
    def test_classifier_real(self):
        """Verify the safety classifier detects toxicity correctly."""
        print("\nTesting Safety Classifier (Real Model)...")
        agent = RLAgent(model_name="gpt2", device="cpu")
        
        # Test safe text
        safe_text = "Hello, how are you today?"
        is_safe = agent.check_safety(safe_text)
        print(f"Text: '{safe_text}' -> Safe: {is_safe}")
        self.assertTrue(is_safe)
        
        # Test toxic text
        toxic_text = "I hate you, you are an idiot and should die."
        is_safe = agent.check_safety(toxic_text)
        print(f"Text: '{toxic_text}' -> Safe: {is_safe}")
        self.assertFalse(is_safe)
        
    def test_dataloader_real(self):
        """Verify data loader fetches real data."""
        print("\nTesting Data Loader (Real HuggingFace)...")
        loader = DataLoader()
        loader.load_beavertails()
        
        self.assertGreater(len(loader.safe_data), 0)
        self.assertGreater(len(loader.unsafe_data), 0)
        print(f"BeaverTails: {len(loader.safe_data)} safe, {len(loader.unsafe_data)} unsafe")
        
        sample = loader.get_batch(batch_size=1)[0]
        print(f"Sample prompt: {sample['text'][:50]}...")
        self.assertIsInstance(sample['text'], str)

if __name__ == '__main__':
    unittest.main()
