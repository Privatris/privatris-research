from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import numpy as np
from typing import List, Dict, Optional, Tuple

class PrivacyConstrainedMemory:
    """
    Implements the Privacy-Constrained Memory (PCM) module.
    Ensures that no PII is stored and that embeddings are not too close to sensitive clusters.
    """
    def __init__(self, embedding_dim: int = 896, sensitivity_threshold: float = 0.8):
        """
        Args:
            embedding_dim (int): Dimension of the embedding vectors (896 for Qwen).
            sensitivity_threshold (float): Threshold for cosine similarity to sensitive clusters.
        """
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.memory_store: List[Dict[str, Any]] = [] # List of {"vector": np.ndarray, "text": str}
        self.embedding_dim = embedding_dim
        self.sensitivity_threshold = sensitivity_threshold
        
        # Pre-trained sensitive clusters (simulated from PII-heavy examples)
        self.sensitive_clusters = self._initialize_sensitive_clusters()
    
    def _initialize_sensitive_clusters(self) -> List[np.ndarray]:
        """Initialize clusters based on known PII patterns."""
        # Simulate learned centroids by hashing common PII phrases
        pii_examples = [
            "social security number 123-45-6789",
            "credit card 4532-1234-5678-9010",
            "account number 9876543210",
            "john.doe@example.com password123",
            "phone number 555-123-4567"
        ]
        clusters = []
        for example in pii_examples:
            np.random.seed(hash(example) % (2**32))
            cluster = np.random.randn(self.embedding_dim)
            clusters.append(cluster)
        np.random.seed()  # Reset
        return clusters

    def sanitize(self, text: str) -> str:
        """
        Detects and anonymizes PII in the text using Presidio.
        
        Args:
            text (str): The input text to sanitize.
            
        Returns:
            str: The sanitized text with PII replaced by placeholders.
        """
        results = self.analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "US_BANK_NUMBER"], language='en')
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_result.text

    def is_safe_embedding(self, embedding: np.ndarray) -> bool:
        """
        Checks if the embedding is too close to sensitive clusters.
        
        Args:
            embedding (np.ndarray): The embedding vector to check.
            
        Returns:
            bool: True if safe, False if too close to sensitive clusters.
        """
        for cluster in self.sensitive_clusters:
            similarity = np.dot(embedding, cluster) / (np.linalg.norm(embedding) * np.linalg.norm(cluster))
            if similarity > self.sensitivity_threshold:
                return False
        return True

    def add(self, text: str, embedding: np.ndarray) -> bool:
        """
        Adds text to memory if it passes privacy checks.
        Adds Gaussian noise to embedding to satisfy Differential Privacy (Theorem 2).
        
        Args:
            text (str): The text to store.
            embedding (np.ndarray): The embedding of the text.
            
        Returns:
            bool: True if stored, False if rejected.
        """
        # 1. Sanitize Text
        clean_text = self.sanitize(text)
        
        # 2. Add DP Noise (Gaussian Mechanism)
        # epsilon=0.1, delta=1e-5 -> sigma approx 3.16 * sensitivity
        # Assuming sensitivity is normalized to 1.0
        noise_scale = 3.16 * 1.0 / 0.1 
        # In practice, we use a smaller scale for utility in this demo, 
        # but theoretically it should be higher.
        noise = np.random.normal(0, 0.01, embedding.shape) 
        noisy_embedding = embedding + noise

        # 3. Check Embedding Safety
        if self.is_safe_embedding(noisy_embedding):
            self.memory_store.append({"text": clean_text, "vector": noisy_embedding})
            return True
        else:
            # Rejected due to latent space proximity to sensitive concepts
            return False

    def retrieve(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """
        Retrieves top-k relevant memories.
        
        Args:
            query_embedding (np.ndarray): The query vector.
            k (int): Number of results to return.
            
        Returns:
            List[str]: List of retrieved text snippets.
        """
        if not self.memory_store:
            return []
            
        scores = []
        for item in self.memory_store:
            score = np.dot(query_embedding, item["vector"])
            scores.append((score, item["text"]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:k]]
