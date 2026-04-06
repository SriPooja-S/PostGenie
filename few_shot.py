import pandas as pd
import json

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class FewShotPosts:
    def __init__(self, file_path="data/processed_posts.json"):
        self.df = None
        self.unique_tags = None
        self.load_posts(file_path)
        
        # Initialize RAG if libraries are available
        self.index = None
        self.encoder = None
        self.rag_enabled = RAG_AVAILABLE
        if self.rag_enabled:
            try:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.build_faiss_index()
            except Exception as e:
                print("Failed to initialize RAG models:", e)
                self.rag_enabled = False

    def load_posts(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            posts = json.load(f)
            self.df = pd.json_normalize(posts)
            self.df['length'] = self.df['line_count'].apply(self.categorize_length)
            # collect unique tags
            all_tags = self.df['tags'].apply(lambda x: x).sum()
            self.unique_tags = list(set(all_tags))

    def get_filtered_posts(self, length, language, tag):
        df_filtered = self.df[
            (self.df['tags'].apply(lambda tags: tag in tags)) &  # Tags contain 'Influencer'
            (self.df['language'] == language) &  # Language is 'English'
            (self.df['length'] == length)  # Line count is less than 5
        ]
        return df_filtered.to_dict(orient='records')

    def categorize_length(self, line_count):
        if line_count < 5:
            return "Short"
        elif 5 <= line_count <= 10:
            return "Medium"
        else:
            return "Long"

    def get_tags(self):
        return self.unique_tags

    def build_faiss_index(self):
        # Create vectors for all posts text
        texts = self.df['text'].tolist()
        embeddings = self.encoder.encode(texts)
        # Initialize FAISS Flat L2 index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def get_similar_posts_by_topic(self, topic, n=2):
        if not self.rag_enabled or self.index is None:
            # Fallback to tag matching if RAG isn't available
            return self.get_filtered_posts("Medium", "English", topic)
        
        # RAG Search
        topic_embedding = self.encoder.encode([topic])
        distances, indices = self.index.search(np.array(topic_embedding), n)
        
        matched_posts = []
        for idx in indices[0]:
            if idx < len(self.df):
                matched_posts.append(self.df.iloc[idx].to_dict())
        return matched_posts


if __name__ == "__main__":
    fs = FewShotPosts()
    # print(fs.get_tags())
    posts = fs.get_filtered_posts("Medium","Hinglish","Job Search")
    print(posts)