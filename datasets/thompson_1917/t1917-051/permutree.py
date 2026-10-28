### PERMUTREE: PERMUTATIONAL REGRESSION ANALYSIS on TREES
### Written by Aki Watanabe
### Contact: awatanabe@bio.fsu.edu
### Department of Biological Science
### Florida State University, Tallahassee, FL

### Program Summary:
### This program takes a NEWICK string file and a CSV
### file of a single continuous variable and conducts
### a permutational regression analysis on phenotypic
### and phylogenetic distances to create a histogram
### of the distribution of regression coefficients.

### Input:
### python prat.py <treefile> <csvfile> <# of iterations>
### Note: To run the program, you will need tree.py module
### to run, which can be downloaded through Peter Beerli's
### website (see README). Additionally, matplotlib and scipy
### Python packages are required.

### Output: 
### regression coefficient and p-value of given data;
### histogram of distribution of coefficients in pdf format;
### p-value of given coefficient

### Future Versions:
### Incorporate error messages.
### Plot of phenotypic and phylogenetic distances w/ regression line.
### Allow multiple NEWICK strings in the tree file.

###### MODULES ######

from itertools import combinations
import math
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from random import shuffle
from scipy import stats
import sys
import tree

###### FUNCTIONS ######

# get_vardist() returns phenotypic distances
def get_vardist(var1, var2):
	return math.fabs((float(var1) - float(var2)))

# get_name() returns list of 'names' to be used in subsequent functions
def get_names(node):
        if not(tree.istip(node)):
            	get_names(node.left)
            	get_names(node.right)
        else:
            	# print node.name
		name_list.append(node.name)

# ancestor() determines whether 'node' is an ancestor of a terminal node
def ancestor(node, name):
	if node.left == -1:
		if node.name == name:
			return True
		else:
			return False	
	left_ancestor = ancestor(node.left, name)
	right_ancestor = ancestor(node.right, name)
	return left_ancestor or right_ancestor

# common_ancestor() returns common ancestors of two terminal nodes
def common_ancestor(node, name1, name2):
	return ancestor(node, name1) and ancestor(node, name2)

# mrca() returns most recent common ancestor of two terminal nodes
def mrca(node, name1, name2):
	if common_ancestor(node.left, name1, name2):
		return mrca(node.left, name1, name2)
	if common_ancestor(node.right, name1, name2):
		return mrca(node.right, name1, name2)
	# print 'name1, name2: ', name1, name2
	return node

# get_phylodist() returns phylogenetic distances for each MRCA
def get_phylodist(my_mrca):
	if my_mrca.left == -1:
		return 0.0
	return 2*my_mrca.left.blength + (get_phylodist(my_mrca.left))

###### PROGRAM ######

f = open(str(sys.argv[1]), 'r')
cv = open(sys.argv[2]).readline()
n = int(sys.argv[3])

newick = str(f.readline())
cont_var = cv.split(',')

### Initialization ###
mytree = tree.Tree()
mytree.myread(newick, mytree.root)
list_vardist = []
name_list = []
name_list.append(get_names(mytree.root))
name_list = name_list[:-1] # For unknown reason, one extra name is added so it is removed here
# print 'name_list', name_list
list_phylodist = []
list_coeff = []
i = 0

### List of variable distances ###
list_var = combinations(cont_var,2)
for var in list_var:
	var1, var2 = var[0], var[1]
	list_vardist.append(get_vardist(var1, var2))
# print 'list_vardist: ', list_vardist

### Regression analysis of observed data ###
comb_name = combinations(name_list, 2)
for pair in comb_name:
	name1, name2 = pair[0], pair[1]
	# print 'pair: ', name1, name2
	my_mrca = mrca(mytree.root, name1, name2)
	# print 'my_mrca: ', my_mrca
	list_phylodist.append(get_phylodist(my_mrca))
	# print 'list phylodist: ', list_phylodist
# print 'final list_phylodist: ', list_phylodist
obs_coeff, obs_intercept, obs_r_value, obs_p_value, obs_std_err = stats.linregress(list_vardist, list_phylodist)
print 'ObsData regression coefficient: ', obs_coeff
print 'ObsData Intercept: ', obs_intercept
print 'ObsData p-value: ', obs_p_value
print 'ObsData R^2: ', obs_r_value

### Permutational regression analysis ###
for itn in range(n):
	list_randvar = []
	shuffle(cont_var) # shuffle() automatically reassigns shuffled values to parameter.
	# print cont_var
	list_randvar = combinations(cont_var,2)
	list_randvardist = []
	for var in list_randvar:
		var1, var2 = var[0], var[1]
		list_randvardist.append(get_vardist(var1, var2))
	# print 'list_randvardist: ', list_randvardist
	rand_coeff, rand_intercept, rand_r_value, rand_p_value, rand_std_err = stats.linregress(list_randvardist, list_phylodist)
	# print 'regression coefficient: ', rand_coeff
	# print 'p value: ', rand_p_value
	list_coeff.append(rand_coeff)
	if math.fabs(float(rand_coeff)) > math.fabs(float(obs_coeff)):
		i = i + 1

### Significance of observed coefficient ###
print 'Randomization Test: ', i/float(n)

### Scatter Plot ###
# matplotlib.scatter(list_randvardist, list_phylodist, s = 20, c='b', marker = 'o', cmap = None, norm = None, vmin = None, vmax = None, alpha = None, linewidths = None, verts = None)

### Distribution Plot ###
fig = plt.figure()
n, bins, patches = plt.hist(list_coeff, 256, normed=1, facecolor = 'green', alpha=0.75)
plt.grid(True)
plt.savefig('permutree_distn.pdf', format = 'pdf');
