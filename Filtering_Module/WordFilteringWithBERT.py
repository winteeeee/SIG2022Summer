from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

model_name = "C:\\Users\\Han SeongMin\\IdeaProjects\\SIG2022Summer\\Filtering_Module\\FilteringModelOutput"

model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=-1,
    return_all_scores=True,
    function_to_apply='sigmoid'
)


def filtering_bert(string):
    for result in pipe(string)[0]:
        if result['label'] == 'clean':
            if result['score'] > 0.5:
                return string

            else:
                return "*"*len(string)
