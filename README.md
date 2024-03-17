# BERT Sentiment Analysis on Amazon Product Review

This project employs the Industrial and Scientific K-core dataset obtained from the Amazon Review Data (2018). This dataset contains reviews, encompassing both ratings and textual content, alongside details about reviewers such as review time, reviewer name, and Unix review time.

The project leverages a pre-trained BERT model to analyze the sentiment of the review text. Following training with the BERT model, the system predicts the sentiment of the test data. To assign sentiment labels, the project transforms product ratings, ranging from 1 to 5, into 0 (Negative), 1 (Neutral), and 2 (Positive).

There are 4 parts in the project:
1. Data Exploration
2. Outlier Removal
3. Data Preprocessing
4. Model Fine-tuning

## 0. Loading Dataset
```python
df = pd.read_json("./Industrial_and_Scientific_5.json.gz", lines=True)

# Unnecessary Columns: reviewTime, style, reviewerName, unixReviewTime, image, vote
df = df.drop(["reviewTime", "style", "reviewerName", "unixReviewTime", "image", "summary", "vote"], axis=1)
```
Load the json file into a dataframe and drop the unnecessary columns.

## 1. Data Exploration
```python
print(df.isna().sum())
df.dropna(subset=['reviewText'], inplace=True)

# Explore data stats
print('Counts:')
print(df.count(), '\n')
print('Averages:')
print(df.mean(numeric_only=True), '\n')
print('Medians:')
print(df.median(numeric_only=True), '\n')
print('Modes:')
print(df.mode(numeric_only=True).iloc[0], '\n')
```
Simply Explore the statistics of the datasets and check whether there is any missing value.

```python
# Display distribution of the number of reviews across products
reviews_across_products = df.groupby('overall')['asin'].count()

review_counts = df['overall'].value_counts().sort_index()
print("Number of reviews for each rating in the sample:")
print(review_counts)

plt.figure(figsize=(6, 4))
plt.bar(reviews_across_products.index, reviews_across_products.values, color='skyblue')
plt.xlabel('Ratings')
plt.ylabel('Number of Products')
plt.title('Distribution of Number of Reviews Across Products')
plt.show()
```
![image](https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/7ce35668-f53c-411f-9267-73d9ab30e96e)

Visualize the distribution of reviews per ratings.

```python
reviews_per_product = df.groupby(['asin', 'overall']).size().unstack(fill_value=0)
reviews_per_product['total_reviews'] = reviews_per_product.sum(axis=1)
print(reviews_per_product)

reviews_per_user = df.groupby(['reviewerID', 'overall']).size().unstack(fill_value=0)
reviews_per_user['total_reviews'] = reviews_per_user.sum(axis=1)
print(reviews_per_user)
```
Display the distribution of the number of reviews per product and the distribution of reviews per user.

```python
df['review_length'] = df['reviewText'].apply(lambda x: len(x))

plt.figure(figsize=(10, 6))
plt.boxplot(df['review_length'], vert=False)
plt.title('Box Plot of Review Lengths')
plt.xlabel('Review Length')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(df['review_length'], vert=False, showfliers=False)
plt.title('Review Lengths without Outliners')
plt.xlabel('Review Length')
plt.grid(True)
plt.show()
```
![image](https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/49901bf6-93ca-4e73-87e3-38ec7b0fa15d)

![image](https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/ee906000-b26c-4b8e-899c-e80154b47bb9)

Display the box plot for lengths of reviews to check outliers and show box plot for lengths of reviews without outliner.

```python
df_review = df["reviewText"]
duplicate_rows = df_review.duplicated()
num_duplicates = duplicate_rows.sum()
duplicate_data = df_review[duplicate_rows]
print("There are", num_duplicates, "duplicate rows in the dataset.")
print(duplicate_data)

print("Drop Duplicates")
df.drop_duplicates(subset=['reviewText'], inplace=True)
num_duplicates = df.duplicated().sum()
print("There are", num_duplicates, "duplicate rows in the dataset.")
```
Check duplicates from the dataset and remove them.

```python
verified_counts = df['verified'].value_counts()
plt.figure(figsize=(6, 4))
verified_counts.plot(kind='bar', color='skyblue')
plt.title('Verified vs Unverified Reviews')
plt.xlabel('Verification')
plt.ylabel('Number of Reviews')
plt.xticks(ticks=[0, 1], labels=['Unverified', 'Verified'], rotation=0)
plt.show()
```
![image](https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/49841444-7a51-4136-b795-e076151f084e)

Visualize the distribution of verified reveiews.

## 2. Data Preprocessing
### Outlier Removal
```python
Q1 = df['review_length'].quantile(0.25)
Q3 = df['review_length'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['review_length'] >= lower_bound) & (df['review_length'] <= upper_bound)]
print(df.shape)

plt.figure(figsize=(10, 6))
plt.boxplot(df['review_length'], vert=False)
plt.title('Box Plot of Review Lengths')
plt.xlabel('Review Length')
plt.grid(True)
plt.show()

sns.boxplot(x='overall', y='review_length', data=df)
plt.title('Boxplot of Review Length for Each Rating (Outliers Removed)')
plt.show()
```
![image](https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/bbea9058-5448-44ff-8f1e-a33fe555888e)

![image](https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/f1269ae5-c098-4d64-acd0-e2c67f4fb324)

Identify the outliers by IQR method and remove them from dataset.

### Data Transformation
```python
df['overall'] = df['overall'].replace({1: 0, 
                                       2: 0, 
                                       3: 1, 
                                       4: 2, 
                                       5: 2})
```
Transform the original labels to 0 (Neagtive), 1 (Netural), 2 (Positive)

### Random Sampling
```python
df_sample = df.sample(n=1000, random_state=2024)
df_sample = df_sample[['reviewText', 'overall']]
df_train, df_val = train_test_split(df_sample, test_size=0.1, random_state=42)

test_df = df[~df.index.isin(df_sample.index)]
test_df_sample = test_df.sample(n=1000, random_state=2024)
df_test = test_df_sample[['reviewText', 'overall']]
```
Randomly select 1000 sample and split it into training set and validation set. 

Randomly select 1000 sample for testing set. 

### Tokenization
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    tokenized_inputs = tokenizer(batch['reviewText'], padding=True, truncation=True, max_length=128, return_tensors='pt')
    tokenized_inputs["labels"] = torch.tensor(batch['overall'])
    return tokenized_inputs

train_dataset = Dataset.from_pandas(df_train).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(df_val).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(df_test).map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
```
Tokenize the data using a BERT tokenizer from a pre-trained model and format the datasets for training, validation, and testing using PyTorch.

This prepares the data for training and evaluating a BERT-based model for later tasks.

## Model Fine-Tuning
```python
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(np.unique(df['overall']))
)
```
Initializes a BERT-based sequence classification model.

```python
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    # Compute precision, recall, and F1-score
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```
Set up evaluation metrics for accuracy, precision, recall and f1score, and training arguments for training a model.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```
Set up a Trainer object for training a model and initiate the training loop, where the model will be trained on the training dataset, evaluated on the evaluation dataset after each epoch.

```python
predictions = trainer.predict(test_dataset)
print(predictions.predictions.shape, predictions.label_ids.shape)

eval_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print("Accuracy:", eval_metrics["accuracy"])
print("Precision:", eval_metrics["precision"])
print("Recall:", eval_metrics["recall"])
print("F1 Score:", eval_metrics["f1_score"])
```
Make predictions on the test dataset using the trained model and compute evaluation metrics based on the predictions and true labels.
The testing accuracy is around 90%.

<img width="400" alt="Screenshot 2024-03-06 at 13 58 17" src="https://github.com/ThomasWongHY/bert_sentiment_analysis/assets/86035047/ff508c66-6888-4d48-b28a-80ab437625c9">


# Reference:
Justifying recommendations using distantly-labeled reviews and fined-grained aspects
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019
https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
