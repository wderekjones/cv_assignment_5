# cv_assignment_5: Pixel Level Classification

This directory contains source to perform a pixel level classification as specified
in the assignment.

* /saved\_models: directory containing the .h5 files of previously trained models
* /report: directory containing this report
* evaluation\_results: directory containing evaluation results
* /baseline: directory containing testing/training data
* \emph{evaluate.py}: contains code related to model evaluation
* \emph{models.py}: contains various helper functions related to model definition and recovery.
* \emph{part2.py}: contains source related to part 2 implementation
* \emph{part3.py}: contains source related to part 3 implementation
* \emph{quick\_evaluate.py}: script that takes command line arguments to model files and evaluates their performance. A second argument specifies whether to print output or write to file.
* \emph{quick\_visualize.py}: script that takes command line arguments to model files and visualizes their performance.
* \emph{train.py}: contains source related to model training.
* \emph{visualize.py}: contains source code related to model performance visualization.

to generate the results for part 2:
> python part2.py

to generate the results for part 3:
> python part3.py

to evaluate the performance of previously trained methods (supply -wtf to write results to file):
> python quick_evaluation.py -m "saved_models/starter_model_p2_balanced_weights.h5"  "saved_models/starter_model_p2_default_weights.h5"  "saved_models/starter_model_p2_reduced_background_weights.h5"

to visualize the performance of previously trained models:
> python quick_visualize.py -m "saved_models/starter_model_p2_balanced_weights.h5"  "saved_models/starter_model_p2_default_weights.h5"  "saved_models/starter_model_p2_reduced_background_weights.h5"
