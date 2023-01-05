from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
)
from sklearn.svm import LinearSVC
from NLP.datasets.dataset import create_dataset_and_extract_features

seed = 9876

def train_model(language, features_options, test_language=None):
    (
        train_features,
        train_labels,
        vectorizer,
        class_map,
    ) = create_dataset_and_extract_features(
        language, features_options, vectorizer=None, class_map=None
    )

    if test_language is not None:
        test_features, test_labels, _, _ = create_dataset_and_extract_features(
            test_language, features_options, vectorizer, class_map
        )

    k_fold = StratifiedKFold(10, shuffle=True, random_state=seed)

    classifiers = [
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed
        ),
        LinearSVC(class_weight="balanced", random_state=seed),
        LogisticRegression(class_weight="balanced", random_state=seed, max_iter=10000)
    ]

    f1_scores = []
    cross_vals = []
    for classifier in classifiers:

        if test_language:
            classifier.fit(train_features, train_labels)
            predicted = classifier.predict(test_features)

        else:
            cross_val = cross_val_score(
                classifier, train_features, train_labels, cv=k_fold, n_jobs=1
            )
            predicted = cross_val_predict(
                classifier, train_features, train_labels, cv=k_fold
            )

            print("Cross val acc", sum(cross_val) / float(len(cross_val)))

            cv_score = sum(cross_val) / float(len(cross_val))
            cross_vals.append(cv_score)

        print(
            confusion_matrix(
                train_labels if not test_language else test_labels, predicted
            )
        )

        gt_labels = train_labels if not test_language else test_labels
        print(f1_score(gt_labels, predicted, average="weighted"))

        fscore = f1_score(gt_labels, predicted, average="weighted")

        f1_scores.append(fscore)

    return f1_scores, cross_vals



if __name__ == '__main__':
    
    option_scores = {}
    for option in [
        "word_ngrams",
        "pos_ngrams",
        "dependency_ngrams",
        "domain_features",
        "doc_len",
    ]:
        # for option in ["pos_ngrams", "dependency_ngrams", "domain_features","doc_len"]:

        print(option)

        for with_domain in [True, False]:
            features_options = {
                "word_ngrams": False,
                "pos_ngrams": False,
                "dependency_ngrams": False,
                "domain_features": False,
                "doc_len": False,
                "language_flag": False,
            }

            features_options[option] = True

            if with_domain:
                features_options["domain_features"] = True

            # f1_scores, cross_vals = train_model('german', features_options, test_language='italian')
            # f1_scores, cross_vals = train_model(
                # "german", features_options, test_language="czech"
            # )

            f1_scores, cross_vals = train_model('czech', features_options)
            # f1_scores, cross_vals = train_model('all', features_options)
            # f1_scores, cross_vals = train_model('italian', features_options)
            # f1_scores, cross_vals = train_model('german', features_options)


            option_new = option if not with_domain else f"{option}_domain"
            option_scores[option_new] = {"f1": f1_scores, "cv": cross_vals}

    import pickle
    with open('cz_results.pickle', 'wb') as handle:
        pickle.dump(option_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    import pdb; pdb.set_trace()

    # 0.5014511974914702 - Logistic DE - doc len
    # 0.5966130620873807 - logistic IT - doc len
    # train_model('german', features_options)

    #            Word   Dep   Pos   Domain  W+d     Dep+d   Pos+d
    # Random -  0.844         0.804
    # Linear -  0.785
    # Logistic - 0.810       0.8191


    # cross lang confusion matrix
    # pos_ngrams - DE - IT
    # [[  3  26   0   0 0]
    #  [  7 321  45   8 0]
    #  [  0  75 226  93 0]

    # pos_ngrams - DE CZ
    # [[0 137  46   5 0]
    #  [0  22  89  54 0]
    #  [0  2  20  59 0]]
    # 0.662235609728027
    # dependency_ngrams
    # [[0 165  22   1 0]
    #  [0 82  66  17 0]
    #  [0 21  33  27 0]]
    # 0.5689418512605513
