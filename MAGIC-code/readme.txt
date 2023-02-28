MAGIC graph generation
======================

Run: python main.py

Explanation: The script does the following:

1. Create a train-set that answers the predicate: lambda a, b, c = > (a and b) or c
2. Create the following test-set:

index,a,b,c,label
0    ,1,1,0,1
1    ,1,0,1,0
2    ,1,1,1,0
3    ,0,0,0,0
4    ,0,1,1,1
5    ,1,0,0,1

3. print to the console the predicate of each sample according to the function reflected by the train-set.
4. Displays a MAGIC graph of 0-labeled samples misclassified as 1 in the test-set.
As you can see, samples 1+2 fit it.

Requirements
============
* graphviz
* sklearn
* pandas
* numpy