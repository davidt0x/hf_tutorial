# -*- coding: utf-8 -*-
from transformers import pipeline

###############################################################################
# Text classification
###############################################################################

# Let's start with one of the most common tasks in NLP: text classification. 
# We need a snippet of text for our models to analyze, so let's use the 
# following (fictious!) customer feedback about a certain online order:

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# While we're at it, let's create a simple wrapper so that we can pretty 
# print out texts
import textwrap

wrapper = textwrap.TextWrapper(width=80, break_long_words=False, 
                               break_on_hyphens=False)
print("Example Text:\n")
print(wrapper.fill(text))
print("\n\n")

# Now suppose that we'd like to predict the _sentiment_ of this text, i.e. 
# whether the feedback is positive or negative. This is a special type of 
# text classification that is often used in industry to aggregate customer 
# feedback across products or services. The example below shows how a 
# Transformer like BERT converts the inputs into atomic chunks called 
# **tokens** which are then fed through the network to produce a single 
# prediction. To load a Transformer model for this task is quite simple. 
# We just need to specify the task in the `pipeline()` function as follows;

sentiment_pipeline = pipeline("text-classification")

# When you run this code, you'll see a message about which Hub model is 
# being used by default. In this case, the `pipeline()` function loads the 
# `distilbert-base-uncased-finetuned-sst-2-english` model, which is a 
# small BERT variant trained on 
# [SST-2](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) 
# which is a sentiment analysis dataset.
#
# The first time you execute the code, the model will be automatically 
# downloaded from the Hub and cached for later use!
# Now we are ready to run our example through pipeline and look at some 
# predictions

print("Sentitment Classificationi:\n")
print(sentiment_pipeline(text))
print("\n\n")

# The model predicts negative sentiment with a high confidence which makes 
# sense given that we have a disgruntled customer. You can also see that the 
# pipeline returns a list of Python dictionaries with the predictions. We can 
# also pass several texts at the same time in which case we would get several 
# dicts in the list for each text one.

# Your turn! Feed a list of texts with different types of sentiment to the 
# `sentiment_pipeline` object. Do the predictions always make sense?

###############################################################################
## Named entity recognition
###############################################################################

# Let's now do something a little more sophisticated. Instead of just finding 
# the overall sentiment, let's see if we can extract **entities** such as 
# organizations, locations, or individuals from the text. This task is called 
# named entity recognition, or NER for short. Instead of predicting just a 
# class for the whole text **a class is predicted for each token**, as shown 
# in the example below:

# Again, we just load a pipeline for NER without specifying a model. This will 
# load a default BERT model that has been trained on the 
# [CoNLL-2003](https://huggingface.co/datasets/conll2003) dataset:

ner_pipeline = pipeline("ner")

# When we pass our text through the model,  we now get a long list of Python 
# dictionaries, where each dictionary corresponds to one detected entity. 
# Since multiple tokens can correspond to a a single entity, we can apply an 
# aggregation strategy that merges entities if the same class appears in 
# consecutive token

entities = ner_pipeline(text, aggregation_strategy="simple")

print("Named Entity Recognition:\n")
for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.2f})")
print("\n\n")

# It seems that the model found most of the named entities but was confused about 
# "Megatron" andn "Decepticons", which are characters in the transformers franchise. 
# This is no surprise since the original dataset probably did not contain many 
# transformer characters. For this reason it makes sense to further fine-tune 
# a model on your on dataset!

###############################################################################
## Question answering
###############################################################################

# In this task, the model is given a **question** and a **context** and needs to 
# find the answer to the question within the context. This problem can be 
# rephrased as a classification problem: For each token the model needs to 
# predict whether it is the start or the end of the answer. In the end we can 
# extract the answer by looking at the span between the token with the highest 
# start probability and highest end probability. You can imagine that this 
# requires quite a bit of pre- and post-processing logic. Good thing that the 
# pipeline takes care of all that! As usual, we load the model by specifying 
# the task in the `pipeline()` function. qa_pipeline = pipeline("question-answering")

outputs = qa_pipeline(question=question, context=text)
question = "What does the customer want?"

# This default model is trained on the famous 
# [SQuAD dataset](https://huggingface.co/datasets/squad). Let's see if we can ask 
# it what the customer wants

print("Question Answering:\n")
print(outputs)
print("\n")

# Awesome, that sounds about right!

###############################################################################
## Text summarization
###############################################################################

# Let's see if we can go beyond these natural language understanding tasks (NLU) 
# where BERT excels and delve into the generative domain. Note that generation 
# is much more computationally demanding since we usually generate one token at 
# a time and need to run this several times. 

summarization_pipeline = pipeline("summarization")

# This model is trained was trained on the 
# [CNN/Dailymail dataset](https://huggingface.co/datasets/cnn_dailymail) to 
# summarize news articles.

outputs = summarization_pipeline(text, max_length=45, clean_up_tokenization_spaces=True)
print("Summarization:\n")
print(wrapper.fill(outputs[0]["summary_text"]))
print("\n\n")

# That's not too bad! We can see the model was able to get the main gist of 
# the customer feedback and even identified the author as "Bumblebee".

###############################################################################
## Translation
###############################################################################

# But what if there is no model in the language of my data? You can still try 
# to translate the text. 
# The [Helsinki NLP team](https://huggingface.co/models?pipeline_tag=translation&sort=downloads&search=Helsinkie-NLP) 
# has provided over 1,000 language pair models for translation 🤯. Here we load 
# one that translates English to German:
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")

outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(wrapper.fill(outputs[0]["translation_text"]))

# We can see that the text is clearly not perfectly translated, but the core 
# meaning stays the same. Another cool application of translation models is 
# data augmentation via backtranslation!

## 7. Zero-shot classification

# As a last example let's have a look at a cool application showing the 
# versatility of transformers: zero-shot classification. In zero-shot 
# classification the model receives a text and a list of candidate labels 
# and determines which labels are compatible with the text. Instead of 
# having fixed classes this allows for flexible classification without 
# any labelled data! Usually this is a good first baseline!
zero_shot_classifier = pipeline("zero-shot-classification")

text = "This is a course about the Transformers library",
classes = ["education", "politics", "business"]

zero_shot_classifier(text, candidate_labels=classes)

classes = ["Treffen", "Arbeit", "Digital", "Reisen"]

outputs = zero_shot_classifier(text, classes, multi_label=True)
print("Zero Shot Classification:\n")
print(outputs)
print("\n\n")

