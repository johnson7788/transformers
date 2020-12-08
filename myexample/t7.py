from transformers import pipeline
nlp = pipeline("ner")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very"
           "close to the Manhattan Bridge which is visible from the window."

print(nlp(sequence))

# [
#     {'word': 'Hu', 'score': 0.9995632767677307, 'entity': 'I-ORG'},
#     {'word': '##gging', 'score': 0.9915938973426819, 'entity': 'I-ORG'},
#     {'word': 'Face', 'score': 0.9982671737670898, 'entity': 'I-ORG'},
#     {'word': 'Inc', 'score': 0.9994403719902039, 'entity': 'I-ORG'},
#     {'word': 'New', 'score': 0.9994346499443054, 'entity': 'I-LOC'},
#     {'word': 'York', 'score': 0.9993270635604858, 'entity': 'I-LOC'},
#     {'word': 'City', 'score': 0.9993864893913269, 'entity': 'I-LOC'},
#     {'word': 'D', 'score': 0.9825621843338013, 'entity': 'I-LOC'},
#     {'word': '##UM', 'score': 0.936983048915863, 'entity': 'I-LOC'},
#     {'word': '##BO', 'score': 0.8987102508544922, 'entity': 'I-LOC'},
#     {'word': 'Manhattan', 'score': 0.9758241176605225, 'entity': 'I-LOC'},
#     {'word': 'Bridge', 'score': 0.990249514579773, 'entity': 'I-LOC'}
# ]