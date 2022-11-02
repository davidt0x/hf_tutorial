from transformers import pipeline
from transformers.utils.hub import hf_cache_home

# Lets do a quick check to see if the hugging face cache home is set
# to something with scratch in the path (\scratch\network\ on adroit,
# \scratch\gpfs on della-gpu).
if "/scratch" not in hf_cache_home:
    raise ValueError(
        f"Your hugging face home directory is set to '{hf_cache_home}'! "
        f"You will run out of space on della-gpu or adroit quickly "
        f"because cached models and datasets are large. Set the environment "
        f"variable HF_HOME to /scratch/network/<NetID>/.cache/huggingface "
        f"adroit or /scratch/gpfs/<NetID>/.cache/huggingface on della-gpu."
    )


# Simply instatiating pipelines will trigger downloading the models
sentiment_pipeline = pipeline("text-classification")
ner_pipeline = pipeline("ner")
summarization_pipeline = pipeline("summarization")
qa_pipeline = pipeline("question-answering")
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
zero_shot_classifier = pipeline(
    "zero-shot-classification", model="vicgalle/xlm-roberta-large-xnli-anli"
)
