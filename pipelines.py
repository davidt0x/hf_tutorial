# -*- coding: utf-8 -*-
"""Getting Started with Transformers.ipynb

Original file is located at
    https://colab.research.google.com/github/huggingface/education-toolkit/blob/main/03_getting-started-with-transformers.ipynb

💡 **Welcome!**

We’ve assembled a toolkit that university instructors and organizers can use to easily prepare labs, homework, or classes. The content is designed in a self-contained way such that it can easily be incorporated into the existing curriculum. This content is free and uses widely known Open Source technologies (`transformers`, `gradio`, etc).

Alternatively, you can request for someone on the Hugging Face team to run the tutorials for your class via the [ML demo.cratization tour](https://huggingface2.notion.site/ML-Demo-cratization-tour-with-66847a294abd4e9785e85663f5239652) initiative!

You can find all the tutorials and resources we’ve assembled [here](https://huggingface2.notion.site/Education-Toolkit-7b4a9a9d65ee4a6eb16178ec2a4f3599).

# Tutorial: Getting Started with Transformers

**Learning goals:** The goal of this tutorial is to learn how:

1. Transformer neural networks can be used to tackle a wide range of tasks in natural language processing and beyond.
3. Transfer learning allows one to adapt Transformers to specific tasks.
2. The `pipeline()` function from the `transformers` library can be used to run inference with models from the [Hugging Face Hub](https://huggingface.co/models).

This tutorial is based on the first of our O'Reilly book [_Natural Language Processing with Transformers_](https://transformersbook.com/) - check it out if you want to dive deeper into the topic!

**Duration**: 30-45 minutes

**Prerequisites:** Knowledge of Python and basic familiarity with machine learning


**Author**: [Lewis Tunstall](https://twitter.com/_lewtun) (feel free to ping me with any questions about this tutorial)

All of these steps can be done for free! All you need is an Internet browser and a place where you can write Python 👩‍💻

## 0. Why Transformers?

Deep learning is currently undergoing a period of rapid progress across a wide variety of domains, including:

* 📖 Natural language processing
* 👀 Computer vision
* 🔊 Audio
* 🧬 Biology
* and many more!

The main driver of these breakthroughs is the **Transformer** -- a novel **neural network** developed by Google researchers in 2017. In short, if you’re into deep learning, you need Transformers!

Here's a few examples of what Transformers can do:

* 💻 They can **generate code** as in products like [GitHub Copilot](https://copilot.github.com/), which is based on OpenAI's family of [GPT models](https://huggingface.co/gpt2?text=My+name+is+Clara+and+I+am).
* ❓ They can be used for **improve search engines**, like [Google did](https://www.blog.google/products/search/search-language-understanding-bert/) with a Transformer called [BERT](https://huggingface.co/bert-base-uncased).
* 🗣️ They can **process speech in multiple languages** to perform speech recognition, speech translation, and language identification. For example, Facebook's [XLS-R model](https://huggingface.co/spaces/facebook/XLS-R-2B-22-16) can automatically transcribe audio in one language to another!

Training these models **from scratch** involves **a lot of resources**: you need large amounts of compute, data, and days to train for 😱.

Fortunately, you don't need to do this in most cases! Thanks to a technique known as **transfer learning**, it is possible to adapt a model that has been trained from scratch (usually called a **pretrained model**), to a variety of downstream tasks. This process is called **fine-tuning** and can typically be carried with a single GPU and a dataset of the size that you're like to find in your university or company.

The models that we'll be looking at in this tutorial are all examples of fine-tuned models, and you can learn more about the transfer learning process in the video below:
"""

from IPython.display import YouTubeVideo

YouTubeVideo("BqqfQnyjmgg")

"""Now, Transformers are coolest kids in town, but how can we use them? If only there was a library that could help us ... oh wait, there is! The [Hugging Face Transformers library](https://github.com/huggingface/transformers) provides a unified API across dozens of Transformer architectures, as well as the means to train models and run inference with them. So to get started, let's install the library with the following command:"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# %pip install transformers[sentencepiece]

"""Now that we've installed the library, let's take a look at some applications!

## 1. Pipelines for Transformers

The fastest way to learn what Transformers can do is via the `pipeline()` function. This function loads a model from the Hugging Face Hub and takes care of all the preprocessing and postprocessing steps that are needed to convert inputs into predictions:

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/pipeline.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=800>

In the next few sections we'll see how these steps are combined for different applications. If you want to learn more about what is happening under the hood, then check out the video below:
"""

YouTubeVideo("1pedAIvTWXk")

"""## 2. Text classification

Let's start with one of the most common tasks in NLP: text classification. We need a snippet of text for our models to analyze, so let's use the following (fictious!) customer feedback about a certain online order:
"""

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

"""While we're at it, let's create a simple wrapper so that we can pretty print out texts:"""

import textwrap

wrapper = textwrap.TextWrapper(width=80, break_long_words=False, break_on_hyphens=False)
print(wrapper.fill(text))

"""Now suppose that we'd like to predict the _sentiment_ of this text, i.e. whether the feedback is positive or negative. This is a special type of text classification that is often used in industry to aggregate customer feedback across products or services. The example below shows how a Transformer like BERT converts the inputs into atomic chunks called **tokens** which are then fed through the network to produce a single prediction:

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/clf_arch.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=600>

To load a Transformer model for this task is quite simple. We just need to specify the task in the `pipeline()` function as follows;
"""

from transformers import pipeline

sentiment_pipeline = pipeline("text-classification")

"""When you run this code, you'll see a message about which Hub model is being used by default. In this case, the `pipeline()` function loads the `distilbert-base-uncased-finetuned-sst-2-english` model, which is a small BERT variant trained on [SST-2](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) which is a sentiment analysis dataset.

💡 The first time you execute the code, the model will be automatically downloaded from the Hub and cached for later use!

Now we are ready to run our example through pipeline and look at some predictions:
"""

sentiment_pipeline(text)

"""The model predicts negative sentiment with a high confidence which makes sense given that we have a disgruntled customer. You can also see that the pipeline returns a list of Python dictionaries with the predictions. We can also pass several texts at the same time in which case we would get several dicts in the list for each text one.

⚡ **Your turn!** Feed a list of texts with different types of sentiment to the `sentiment_pipeline` object. Do the predictions always make sense?

## 3. Named entity recognition

Let's now do something a little more sophisticated. Instead of just finding the overall sentiment, let's see if we can extract **entities** such as organizations, locations, or individuals from the text. This task is called named entity recognition, or NER for short. Instead of predicting just a class for the whole text **a class is predicted for each token**, as shown in the example below:

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/ner_arch.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=600>

Again, we just load a pipeline for NER without specifying a model. This will load a default BERT model that has been trained on the [CoNLL-2003](https://huggingface.co/datasets/conll2003) dataset:
"""

ner_pipeline = pipeline("ner")

"""When we pass our text through the model,  we now get a long list of Python dictionaries, where each dictionary corresponds to one detected entity. Since multiple tokens can correspond to a a single entity ,we can apply an aggregation strategy that merges entities if the same class appears in consequtive tokens:"""

entities = ner_pipeline(text, aggregation_strategy="simple")
print(entities)

"""This isn't very easy to read, so let's clean up the outputs a bit:"""

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.2f})")

"""That's much better! It seems that the model found most of the named entities but was confused about "Megatron" andn "Decepticons", which are characters in the transformers franchise. This is no surprise since the original dataset probably did not contain many transformer characters. For this reason it makes sense to further fine-tune a model on your on dataset!

Now that we've seen an example of text and token classification using Transformers, let's look at an interesting application called **question answering**.

## 4. Question answering

In this task, the model is given a **question** and a **context** and needs to find the answer to the question within the context. This problem can be rephrased as a classification problem: For each token the model needs to predict whether it is the start or the end of the answer. In the end we can extract the answer by looking at the span between the token with the highest start probability and highest end probability:

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/qa_arch.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=600>

You can imagine that this requires quite a bit of pre- and post-processing logic. Good thing that the pipeline takes care of all that! As usual, we load the model by specifying the task in the `pipeline()` function:
"""

qa_pipeline = pipeline("question-answering")

"""This default model is trained on the famous [SQuAD dataset](https://huggingface.co/datasets/squad). Let's see if we can ask it what the customer wants:"""

question = "What does the customer want?"

outputs = qa_pipeline(question=question, context=text)
outputs

"""Awesome, that sounds about right!

## 5. Text summarization

Let's see if we can go beyond these natural language understanding tasks (NLU) where BERT excels and delve into the generative domain. Note that generation is much more computationally demanding since we usually generate one token at a time and need to run this several times. An example for how this process works is shown below:

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/gen_steps.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=600>

A popular task involving generation is summarization. Let's see if we can use a transformer to generate a summary for us:
"""

summarization_pipeline = pipeline("summarization")

"""This model is trained was trained on the [CNN/Dailymail dataset](https://huggingface.co/datasets/cnn_dailymail) to summarize news articles."""

outputs = summarization_pipeline(text, max_length=45, clean_up_tokenization_spaces=True)
print(wrapper.fill(outputs[0]["summary_text"]))

"""That's not too bad! We can see the model was able to get the main gist of the customer feedback and even identified the author as "Bumblebee".

## 6. Translation

But what if there is no model in the language of my data? You can still try to translate the text. The [Helsinki NLP team](https://huggingface.co/models?pipeline_tag=translation&sort=downloads&search=Helsinkie-NLP) has provided over 1,000 language pair models for translation 🤯. Here we load one that translates English to German:
"""

translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")

"""Let's translate the our text to German:"""

outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(wrapper.fill(outputs[0]["translation_text"]))

"""We can see that the text is clearly not perfectly translated, but the core meaning stays the same. Another cool application of translation models is data augmentation via backtranslation!

## 7. Zero-shot classification

As a last example let's have a look at a cool application showing the versatility of transformers: zero-shot classification. In zero-shot classification the model receives a text and a list of candidate labels and determines which labels are compatible with the text. Instead of having fixed classes this allows for flexible classification without any labelled data! Usually this is a good first baseline!
"""

zero_shot_classifier = pipeline(
    "zero-shot-classification", model="vicgalle/xlm-roberta-large-xnli-anli"
)

"""Let's have a look at an example:"""

text = (
    "Dieser Tutorial ist großartig! Ich hoffe, dass jemand von Hugging Face meine"
    " Universität besuchen wird :)"
)
classes = ["Treffen", "Arbeit", "Digital", "Reisen"]

zero_shot_classifier(text, classes, multi_label=True)

"""This seems to have worked really well on this short example. Naturally, for longer and more domain specific examples this approach might suffer.

## 8. Going beyond text

As mentioned at the start of this tutorial, Transformers can also be used for domains other than NLP! For these domains, there are many more pipelines that you can experiment with. Look at the following list for an overview:
"""

from transformers import pipelines

for task in pipelines.SUPPORTED_TASKS:
    print(task)

"""Let's have a look at an application involving images!

### Computer vision

Recently, transformer models have also entered computer vision. Check out the DETR model on the [Hub](https://huggingface.co/facebook/detr-resnet-101-dc5):

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/object_detection.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=400>

### Audio

Another promising area is audio processing. Especially Speech2Text there have been some promising advancements recently. See for example the [wav2vec2 model](https://huggingface.co/facebook/wav2vec2-base-960h):

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/speech2text.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=400>

### Table QA

Finally, a lot of real world data is still in form of tables. Being able to query tables is very useful and with [TAPAS](https://huggingface.co/google/tapas-large-finetuned-wtq) you can do tabular question-answering:

<img src="https://github.com/huggingface/workshops/blob/main/nlp-zurich/images/tapas.png?raw=1" alt="Alt text that describes the graphic" title="Title text" width=400>

## 9. Where to next?

Hopefully this tutorial has given you a taste of what Transformers can do and you're now excited to learn more! Here's a few resources you can use to dive deeper into the topic and the Hugging Face ecosystem:

🤗 **A Tour through the Hugging Face Hub**

In this tutorial, you get to:
- Explore the over 30,000 models shared in the Hub.
- Learn efficient ways to find the right model and datasets for your own task.
- Learn how to contribute and work collaboratively in your ML workflows

***Duration: 20-40 minutes***

👉 [click here to access the tutorial](https://www.notion.so/Workshop-A-Tour-through-the-Hugging-Face-Hub-2098e4bae9ba4288857e85c87ff1c851)

✨ **Build and Host Machine Learning Demos with Gradio & Hugging Face**

In this tutorial, you get to:
- Explore ML demos created by the community.
- Build a quick demo for your machine learning model in Python using the `gradio` library
- Host the demos for free with Hugging Face Spaces
- Add your demo to the Hugging Face org for your class or conference

***Duration: 20-40 minutes***

👉 [click here to access the tutorial](https://colab.research.google.com/github.com/huggingface/education-toolkit/tree/main/02_ml-demos-with-gradio.ipynb)

🎓 **The Hugging Face Course**

This course teaches you about applying Transformers to various tasks in natural language processing and beyond. Along the way, you'll learn how to use the Hugging Face ecosystem — 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers, and 🤗 Accelerate — as well as the Hugging Face Hub. It's completely free too!
"""

YouTubeVideo("00GKzGyWFEs")
