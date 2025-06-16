# EnDeep4mC: A dual-adaptive feature encoding framework in deep ensembles for predicting DNA N4-methylcytosine sites

## data/4mC
The dataset we used in this experiment is derived from the 4mC dataset constructed in the work of EpiTEAmDNA.

For details, refer to (Li F, Liu S, Li K, Zhang Y, Duan M, Yao Z, Zhu G, Guo Y, Wang Y, Huang L, Zhou F. EpiTEAmDNA: Sequence feature representation via transfer learning and ensemble learning for identifying multiple DNA epigenetic modification types across species. Comput Biol Med. 2023 Jun;160:107030. doi: 10.1016/j.compbiomed.2023.107030. Epub 2023 May 11. PMID: 37196456.)

## evaluations
Evaluation results of various experiments.
1) ablation_feature: Design and results of feature ablation experiments.
2) ablation_model_indiv: Design and results of model ablation experiments (using independent test sets).
3) acc_heatmap: Accuracy results of 6 species trained on the base model (represented by heatmaps).
4) acc_matrix: Accuracy results of 6 species trained on the base model.
5) acc_matrix_extra: Accuracy results of 10 supplementary species trained on the base model.
6) cross_predict: Results of cross-species prediction experiment.
7) encoding_methods_performance: Prediction performance on independent test sets of all 16 species after encoding by 14 feature encoding methods respectively.
8) feature_compare: Performance comparison of EnDeep4mC before and after feature selection.
9) kmer_analysis: Design and Results of kmer spectrum analysis experiment for eukaryotes/prokaryotes.
10) visualization: Visualizing the above experimental results.
11) workflow_plot: Some images in overall workflow.

## feature_engineering
The feature engineering module, which can be transferred to the feature selection & encoding process of other deep learning models.
1) fea_index.py: Performance quantification functions for 14 features across 6 species.
2) fea_index_extra.py: Performance quantification functions for 14 features across 10 supplementary species.
3) feature_selection.py: Feature selection support functions (6 species).
4) feature_selection_extra.py: Feature selection support functions (10 supplementary species).
5) ifs_on_base_models.py: Species-model joint incremental feature selection function (6 species).
6) ifs_on_base_models_extra.py: Species-model joint incremental feature selection function (10 supplementary species).
7) fea_index: Stores the results of running fea_index.py.
8) fea_index_extra_species: Stores the results of running fea_index_extra.py.
9) ifs_result: Stores the results of running ifs_on_base_models.py.
10) ifs_result_extra_species: Stores the results of running ifs_on_base_models_extra.py.

## fs
Contains detailed definitions of several feature encoding methods from the biological tool iLearn. We mainly referenced the definitions of 14 candidate feature encoding methods from the open-source code of EnDeep4mC mentioned above.

## models
Definitions of the deep learning models proposed in this study.
1) deep_models: Contains definitions of 3 deep learning base classifiers.
2) pretrain_ensemble_model_5cv: Definitions and pre-training scripts for stacked ensemble models (5-fold cross-validation).
3) pretrain_ensemble_model_indiv: Definitions and pre-training scripts for stacked ensemble models (independent test set validation).

## prepare
Definitions for pre-training base models. The files prepare_dl.py and prepare_ml.py are from the open-source code of the EnDeep4mC work. We referenced the data processing functions for DNA sequences.
1) prepare_dl.py: A configuration file related to deep learning in the EnDeep4mC project. This experiment mainly uses some of its data processing functions.
2) prepare_ml.py: A configuration file related to machine learning in the EnDeep4mC project. This experiment mainly uses some of its data processing functions.
3) pretrain_base_models_5cv: Pre-training scripts for base models with dynamic feature selection strategies across 6 species (5-fold cross-validation).
4) pretrain_base_models_indiv: Pre-training scripts for base models with dynamic feature selection strategies across 6 species (independent test set validation).
5) evaluate_all_species: Pre-training scripts using all 14 encodings across 6 species (independent test set validation).
6) evaluate_extra_species: Pre-training scripts for base models with dynamic feature selection strategies across 10 supplementary species (independent test set validation).

## pretrained_models
Stores pre-trained models (.h5) from the prepare and models modules.
1) 5cv: Stores all pre-trained models using 5-fold cross-validation.
2) indiv: Stores all pre-trained models using independent test set validation.

## tools
Some supplementary tools involved in this study, only used for evaluation testing and auxiliary analysis, not actually used in our work.
1) CD-HIT.py: Script for removing DNA sequence redundancy using the CD-HIT tool.
2) GC_content.py: Script for evaluating the GC content in DNA sequences.
3) diversity_metrics.py: Script for evaluating the diversity metrics of base models.
4) motif.py: Script for finding motifs in DNA sequences using the meme tool.
5) tools.py: Originally supplementary tools in the EnDeep4mC project, including functions such as t-sne visualization.

## web_server
A web server built based on the proposed EnDeep4mC model, which can be used online by users.
