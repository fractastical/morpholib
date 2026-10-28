PERMUTREE: PERMUTATIONAL REGRESSION ANALYSIS on TREES

Written by Aki Watanabe
Contact: awatanabe@bio.fsu.edu (awatanabe@amnh.org)
Department of Biological Science
Florida State University, Tallahassee, FL

SUMMARY

This program takes a NEWICK string file and a CSV file of a single continuous variable and conducts a permutational regression analysis on phenotypic and phylogenetic distances to create a histogram of the distribution of regression coefficients. Please note that this program was written in Python version 2.7, thus, the print command is not a proper function. To run the script in later versions, modify “print <x>” to “print(<x>)”.

To run: python permutree.py <treefile> <csvfile> <# of iterations>

NB: To run the program, you will need tree.py module to run, which can be downloaded through Peter Beerli's website <http://people.sc.fsu.edu/~pbeerli/classes/isc5317-notes/isc5317_library/introduction/>. You will also need matplotlib and scipy Python packages to run this analysis. 


OUTPUT
 
1. Console: regression coefficient and p-value of given data.
2. Console: p-value of observed coefficient relative to distribution.
3. permutree_distn.pdf: histogram of distribution of coefficients in pdf format.