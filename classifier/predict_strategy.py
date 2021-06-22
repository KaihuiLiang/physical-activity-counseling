def predict_strategy():
    test_data_path = "test"
    basemodel = "./models/strategy"
    num_labels = 13

    featured_documents, raw_documents, document_names = get_documents(test_data_path)

    raw_doc = raw_documents[0]
    document_name = document_names[0]
    featured_doc = featured_documents[0]  

    pd.set_option('display.max_columns', None)

    model = BertForSequenceClassification.from_pretrained(
        basemodel,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    model.eval()

    for i , doc in enumerate(featured_documents):
        print("predicting doc: ", i)

        featured_doc = featured_documents[i]
        document_name = document_names[i]
        raw_doc = raw_documents[i]

        final_result = predict_document(featured_doc, model)

        raw_doc["strategy"] = final_result

        # print("raw_doc", raw_doc)
        raw_doc.to_csv("./output/{}.csv".format(document_name))


def get_documents(test_data_path):
    category = 'strategy'
    context = 10 

    full_dataset = MPedDataset(get_tokenizer(), annotated=True, category=category, context_width=context, block_size=256)
    document_names = MPedDataset.documents(test_data_path, "testset")

    featured_datasets = []
    raw_datasets = []
    for doc_name in document_names:

        dataset = MPedFromExample(full_dataset.get_samples([doc_name]))
        featured_datasets.append(dataset)

        raw_data = full_dataset.get_data([doc_name])
        raw_data = raw_data[["dataset", "document", "row", "speaker", "pos_in_doc", "context", "utterance"]].copy()
        raw_datasets.append(raw_data)

    return featured_datasets, raw_datasets, document_names
  
 if __name__ == '__main__':
  predict_strategy()
  
