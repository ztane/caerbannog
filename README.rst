Rabbit of Caerbannog
--------------------

..

    Well, that's no ordinary rabbit - that's the most foul, cruel, and bad-tempered rodent you ever set eyes on!

    -- Tim the Enchanter

This module is a high-level interface for the Vowpal Wabbit machine learning system. Currently it relies on 
the `wabbit_wappa` module for lower-level interaction, but strives to provide a more high-level object-oriented interface.

There are currently 3 kinds of `Rabbit`s you can `import from` `caerbannog`:

`Rabbit`
    Your standard rabbit instance. By default runs Vowpal Wabbit using pipes for stdin/stdout

`ActiveRabbit`
    Runs Vowpal Wabbit in active learning mode, using TCP socket

`OfflineRabbit`
    The initializer expects the argument `fp` which is an open file with `'wt'` mode.
    the inputs fed to `teach` will be written to this file for offline processing.


Movie Review Sentiments - Active learning demo with caerbannog
--------------------------------------------------------------

Import relevant modules here. We are using the ActiveRabbit for active
online learning

.. code:: python

    from caerbannog import ActiveRabbit, Example
    from itertools import islice
    import random
    import nltk
    from nltk.corpus import movie_reviews
Create an active bunny, with active mellowness of 0.01

.. code:: python

    rabbit = ActiveRabbit(loss_function='logistic', active_mellowness=0.01)
    rabbit.start()
Load the documents from NLTK movie review corpus (note that you need to
download these first by nltk.download(). For each document, make a tuple
(document\_words, category) where category is either 'pos' or 'neg' and
document\_words is a list of words from tokenizer.

.. code:: python

    documents = [(list(movie_reviews.words(fileid)), category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    len(documents)



.. parsed-literal::

    2000



The feature extractor function. First filters out all tokens that are
non-alphanumeric. Then make the 'w' namespace consist of all the words
in the review; 'n' consists of 2-4 ngrams of the document words.

.. code:: python

    def document_features(document_words):
        document_words = list(filter(str.isalnum, document_words))
        example = Example()
        example['w'].add_features(set(document_words))
        ngrams = set()
        for j in range(2, 5):
            ngrams.update('_'.join(i) for i in nltk.ngrams(document_words, j))
    
        example['n'].add_features(ngrams)
        return example
Vowpal Wabbit expects labels to be -1 and 1 for logistic binary
classifier

.. code:: python

    def convert_sent(sent):
        return {'pos': 1, 'neg': -1}[sent]
Convert the sentiment value and extract features.

.. code:: python

    examples = [ (convert_sent(sent), document_features(doc)) for (doc, sent) in documents ]
Train with 1500 first examples and keep the remaining ones for
verification

.. code:: python

    teach, test = examples[:1500], examples[1500:]
Teach the filter. We ask for prediction for each example; if the
importance is over 1 we "label" the example and teach it to the
classifier. We repeat the classification 40 times to ensure that the
classifier has had enough to adjust the weights.

.. code:: python

    taught = 0
    predicted = 0
    labelled = set()
    for i in range(10):
        for sent, ex in teach:
            predicted += 1
            if rabbit.predict(example=ex).importance >= 1:
                rabbit.teach(label=sent, example=ex)
                taught += 1
                labelled.add(ex)
    
    print("Predicted {}, taught {} (ratio {}). {} unique inputs labelled"
          .format(predicted, taught, taught/predicted, len(labelled)))

.. parsed-literal::

    Predicted 15000, taught 1057 (ratio 0.07046666666666666). 1042 unique inputs labelled


Test with the testing set. For each correctly labelled example, increase
the counter

.. code:: python

    correct = 0
    for sent, ex in test:
        prediction = rabbit.predict(example=ex)
        if prediction.label == sent:
            correct += 1
            
    print("{} inputs predicted. {} correct; ratio {}".format(len(test), correct, correct / len(test)))

.. parsed-literal::

    500 inputs predicted. 418 correct; ratio 0.836


License
-------

MIT license.

