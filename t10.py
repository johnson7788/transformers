import nlp

# Print all the available datasets
res = nlp.list_datasets()

# Load a dataset and print the first examples in the training set
squad_dataset = nlp.load_dataset('squad')
print(squad_dataset['train'][0])

# List all the available metrics
print(nlp.list_metrics())

# Load a metric
squad_metric = nlp.load_metric('squad')