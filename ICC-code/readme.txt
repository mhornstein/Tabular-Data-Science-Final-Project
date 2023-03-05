ICC graph generation
====================

Run: python main.py

Explanation: main.py contains 5 datasets for visualization (by default the 5th dataset [Stratification] is enabled).
You can comment it and uncomment another dataset visualization (according to the comments) to browse the different resulting ICC graph.

Illustrating the Iris dataset
=============================

Run: python Iris_Ilustrations.py

Explanation: This will generate an illustration like in Figure 1 in the report.
You can play with max_depth parameter in code line 48 to get different splits count.

Requirements
============
* graphviz
* sklearn
* pandas
* numpy