from datasets import load_dataset
from transformers import BertForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, TextClassificationPipeline
from sklearn.metrics import label_ranking_average_precision_score
from torch import torch
import tqdm
from sklearn.metrics import classification_report
from transformers.pipelines.base import KeyDataset


dataset = load_dataset('smilegate-ai/kor_unsmile')
print(dataset["train"][0])
unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]

model_name = 'beomi/kcbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    tokenized_examples = tokenizer(str(examples["문장"]))
    tokenized_examples['labels'] = torch.tensor(examples["labels"], dtype=torch.float)
    return tokenized_examples


tokenized_dataset = dataset.map(preprocess_function)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask', 'token_type_ids'])
print(tokenized_dataset['train'][0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
num_labels=len(unsmile_labels) # Label 갯수

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.config.id2label = {i: label for i, label in zip(range(num_labels), unsmile_labels)}
model.config.label2id = {label: i for i, label in zip(range(num_labels), unsmile_labels)}

print(model.config.label2id)
print(model.config)


def compute_metrics(x):
    return {
        'lrap': label_ranking_average_precision_score(x.label_ids, x.predictions),
    }


batch_size = 64
args = TrainingArguments(
    output_dir="FilteringModelOutput",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='lrap',
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)
#trainer.train()
#trainer.save_model()

pipe = TextClassificationPipeline(
    model = model,
    tokenizer = tokenizer,
    device=-1,
    return_all_scores=True,
    function_to_apply='sigmoid'
)

for result in pipe("이래서 여자는 게임을 하면 안된다")[0]:
    print(result)


def get_predicated_label(output_labels, min_score):
    labels = []
    for label in output_labels:
        if label['score'] > min_score:
            labels.append(1)
        else:
            labels.append(0)
    return labels
predicated_labels = []

for out in tqdm.tqdm(pipe(KeyDataset(dataset['valid'], '문장'))):
    predicated_labels.append(get_predicated_label(out, 0.5))
print(classification_report(dataset['valid']['labels'], predicated_labels))