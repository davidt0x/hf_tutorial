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
# The specification of models can be left off, however, in production
# this is not a good practice as the default models can change and
# unexpected behaviour can occur. It is also best to specify a model
# revision has well.
sentiment_pipeline = pipeline(
    "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-cased-distilled-squad"
)
summarization_pipeline = pipeline(
    "summarization", model="sshleifer/distilbart-cnn-12-6"
)
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
zero_shot_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)
