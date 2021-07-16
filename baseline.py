from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from data import read_train_xlsx


def main():
    train_texts, train_labels, test_texts, test_labels = read_train_xlsx(is_tuning_hyper_parameters=True)
    
    baseline_model = make_pipeline(CountVectorizer(ngram_range=(1, 3)), LogisticRegression()).fit(train_texts, train_labels)
    baseline_predicted = baseline_model.predict(test_texts)
    print(classification_report(test_labels, baseline_predicted))
    """
                  precision    recall  f1-score   support
    
        其他旅游      0.72      0.21      0.33       343
        其他医疗      0.78      0.21      0.34       294
        其他文艺      1.00      0.29      0.45       126
        体育         0.95      0.46      0.62       446
        军事         0.82      0.21      0.34       330
        娱乐         0.89      0.32      0.47       470
        房产         0.67      0.26      0.38       313
        教育         0.93      0.38      0.54       290
        汽车         0.16      0.97      0.28       467
        游戏         1.00      0.77      0.87       377
        科技         0.76      0.32      0.45       413
        财经         0.73      0.33      0.45       489
    
        accuracy                           0.42      4358
       macro avg       0.78      0.39      0.46      4358
    weighted avg       0.76      0.42      0.46      4358
    """


if __name__ == '__main__':
    main()
