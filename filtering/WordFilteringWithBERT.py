from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

model_name = 'FilteringModelOutput'

model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=-1,
    return_all_scores=True,
    function_to_apply='sigmoid'
)


def filtering(string):
    for result in pipe(string)[0]:
        if result['label'] == 'clean':
            if result['score'] > 0.5:
                return string

            else:
                return "*"*len(string)
