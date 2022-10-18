from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier('We are very happy to introduce pipeline to the transformers repository.'))
