model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=41)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
