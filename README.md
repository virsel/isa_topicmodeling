# Innovation Strength Analysis with Patents - Topic Modeling

This project uses BERTopic to perform topic modeling on patent embeddings and texts. It generates an overview of contained topics and assigns patents to specific topics based on keyword similarity.

## Project Structure

- `topic_modeling.ipynb`: Main analysis notebook
- `topic2doc.ipynb`: Topic assignment notebook

## Topic Modeling Process

1. Analyze patent embeddings and texts using BERTopic
2. Generate an overview of contained topics
3. Define 2 hypernyms
4. Declare 8 topics for each hypernym
5. Assign keywords to topics 
6. Assign topics to docs based on cosine similarity between keywords and patent text embeddings


## Requirements

- Python 3.8
- Jupyter Notebook
- Other dependencies see requirements.txt

## Usage

1. Run `topic_modeling.ipynb` to perform the initial topic modeling and keyword assignment.
2. Run `topic2doc.ipynb` to assign topics to individual patents.