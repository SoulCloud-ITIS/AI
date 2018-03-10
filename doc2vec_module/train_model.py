from gensim.models import Doc2Vec


def get_model_for_genre(documents : list) -> Doc2Vec:
    model = Doc2Vec(documents, dm=0, alpha=0.025, size=20, min_alpha=0.025, min_count=0)

    for epoch in range(2):
        print('Эпоха #{0}'.format(epoch))
        token_count = sum([len(document) for document in documents])
        model.train(documents, total_examples=token_count, epochs=model.iter)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model
