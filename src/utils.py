import dill

class EmbedsReduced:
  """ Use this for pre-calculated reduced embeddings """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings

  def fit(self, X):
    return self

  def transform(self, X):
    return self.reduced_embeddings

def create_preumap(reduced_embeddings):
    return EmbedsReduced(reduced_embeddings)
  

def custom_save(topic_model, file_path):
    # Save each component separately
    components = {
        "embedding_model": topic_model.embedding_model,
        "hdbscan_model": topic_model.hdbscan_model,
        "umap_model": topic_model.umap_model,
        "ctfidf_model": topic_model.ctfidf_model,
        "vectorizer_model": topic_model.vectorizer_model,
        
    }
    
    representation_model =  topic_model.representation_model
    
    # Temporarily set components to None in the topic model
    topic_model.embedding_model = None
    topic_model.umap_model = None
    topic_model.ctfidf_model = None
    topic_model.vectorizer_model = None
    topic_model.representation_model = None
    
    # Serialize the topic model
    with open(file_path, "wb") as f:
        dill.dump(topic_model, f)

    # Save components separately
    for name, component in components.items():
        if component is not None:
            with open(f"{file_path}_{name}.pkl", "wb") as f:
                dill.dump(component, f)
    
    # Restore components in the topic model
    topic_model.embedding_model = components["embedding_model"]
    topic_model.umap_model = components["umap_model"]
    topic_model.ctfidf_model = components["ctfidf_model"]
    topic_model.vectorizer_model = components["vectorizer_model"]
    topic_model.representation_model = representation_model