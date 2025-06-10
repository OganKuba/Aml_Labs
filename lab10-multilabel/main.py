# task1.py
from data_loader import load_emotions_dataset
from models import (
    train_binary_relevance,
    predict_binary_relevance,
    train_classifier_chain,
    predict_classifier_chain,
    train_ecc,
    predict_ecc
)
from evaluation import subset_accuracy, hamming_score
from utils import print_results

def main():
    # Load Data
    X_train, X_test, Y_train, Y_test = load_emotions_dataset()

    # Binary Relevance
    br_models = train_binary_relevance(X_train, Y_train)
    Y_pred_br = predict_binary_relevance(br_models, X_test)
    br_subset_acc = subset_accuracy(Y_test, Y_pred_br)
    br_hamming = hamming_score(Y_test, Y_pred_br)
    print_results("Binary Relevance", br_subset_acc, br_hamming)

    # Classifier Chain (default order)
    cc_model = train_classifier_chain(X_train, Y_train)
    Y_pred_cc = predict_classifier_chain(cc_model, X_test)
    cc_subset_acc = subset_accuracy(Y_test, Y_pred_cc)
    cc_hamming = hamming_score(Y_test, Y_pred_cc)
    print_results("Classifier Chain", cc_subset_acc, cc_hamming)

    # Ensemble of Classifier Chains
    ecc_models = train_ecc(X_train, Y_train, n_chains=5)
    Y_pred_ecc = predict_ecc(ecc_models, X_test)
    ecc_subset_acc = subset_accuracy(Y_test, Y_pred_ecc)
    ecc_hamming = hamming_score(Y_test, Y_pred_ecc)
    print_results("Ensemble of Classifier Chains", ecc_subset_acc, ecc_hamming)

if __name__ == "__main__":
    main()
