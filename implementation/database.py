# core/database.py
from elasticsearch import Elasticsearch
import numpy as np
import config

class ESClient:
    def __init__(self):
        self.es = Elasticsearch(config.ES_HOST)
        self.index = config.ES_INDEX
        
    def store_embeddings(self, name, embeddings):
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        body = {
            "person_name": name,
            "embedding": avg_embedding,
            "timestamp": "now"
        }
        self.es.index(index=self.index, document=body)
    
    def search_face(self, embedding, threshold=0.85):
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        }
        response = self.es.search(
            index=self.index,
            query=script_query,
            size=1
        )
        if not response['hits']['hits']:
            return None
        best_match = response['hits']['hits'][0]
        return best_match if best_match['_score'] >= threshold else None