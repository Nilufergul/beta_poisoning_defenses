#!/bin/bash
if [[ $1 == "mnist_ablation" ]]; then
  python src/experiments/mnist/run_ablation.py

elif [[ $1 == "cifar_ablation" ]]; then
  python src/experiments/cifar10/run_ablation.py

elif [[ $1 == "mnist_bin" ]]; then
  python src/experiments/mnist/run_pairs.py --generator="beta" --ds="4-0" --n_proto=15 --classifier="svm"
  python src/experiments/mnist/run_pairs.py --generator="beta" --ds="4-0" --n_proto=15 --classifier="ridge"
  #python src/experiments/mnist/run_pairs.py --generator="beta" --ds="4-0" --n_proto=15 --classifier="logistic"
  #python src/experiments/mnist/run_pairs.py --generator="beta" --ds="4-0" --n_proto=15 --classifier="naive_bayes"
  #python src/experiments/mnist/run_pairs.py --generator="beta" --ds="4-0" --n_proto=15 --classifier="decision_tree"
  #python src/experiments/mnist/run_pairs.py --generator="beta" --ds="4-0" --n_proto=15 --classifier="lda"

elif [[ $1 == "cifar_bin" ]]; then
  python src/experiments/cifar10/run_pairs.py --generator="beta flip white" --ds="6-8" --n_proto=30 --classifier="svm"
  python src/experiments/cifar10/run_pairs.py --generator="beta flip white" --ds="6-8" --n_proto=30 --classifier="logistic"

  python src/experiments/cifar10/run_pairs.py --generator="beta flip white" --ds="7-8" --n_proto=30 --classifier="svm"
  python src/experiments/cifar10/run_pairs.py --generator="beta flip white" --ds="7-8" --n_proto=30 --classifier="logistic"

elif [[ $1 == "mnist_triplet" ]]; then
  python src/experiments/mnist/run_triplet.py --generator="beta flip" --n_proto=15 --classifier="svm" --ds="9-4-0"
  python src/experiments/mnist/run_triplet.py --generator="beta flip" --n_proto=15 --classifier="svm" --ds="3-7-5"
fi