""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
A wrapper for Random Tree regression.  		  	   		   	 		  		  		    	 		 		   		 		  
	  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
from scipy import stats

  		  	   		   	 		  		  		    	 		 		   		 		  
class RTLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a RT Learner.
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self,leaf_size, verbose=False):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.tree = np.array([], dtype='object')
        self.leaf_size = leaf_size
  		  	   		   	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "mghoneim3"  # replace tb34 with your Georgia Tech username

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size:
            mode_y = stats.mode(data_y, axis=None)[0][0]
            return np.array([['leaf', mode_y, np.nan, np.nan]])
        elif len(np.unique(data_y)) == 1:
            return np.array([['leaf', data_y[0], np.nan, np.nan]])
        else:
            i = np.random.randint(0, data_x.shape[1]-1)
            split_val = np.median(data_x[:, i])
            if len(data_x[data_x[:, i] <= split_val]) == 0 or len(data_x[data_x[:, i] > split_val]) == 0:
                mode_y = stats.mode(data_y, axis=None)[0][0]
                return np.array([['leaf', mode_y, np.nan, np.nan]])

            left_tree = self.build_tree(data_x[data_x[:, i] <= split_val], data_y[data_x[:, i] <= split_val])
            right_tree = self.build_tree(data_x[data_x[:, i] > split_val], data_y[data_x[:, i] > split_val])
            root = np.array([[i, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        self.tree = self.build_tree(data_x, data_y)



    def query(self, xtest):
        """
        Estimate a set of test points given the model we built.
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        y_pred = np.zeros(xtest.shape[0])
        for j in range(len(y_pred)):

            split_attr = self.tree[0, 0]
            split_val = self.tree[0, 1]

            i = 0
            while True:

                if split_attr == 'leaf':
                    y_pred[j] = float(split_val)
                    break

                if xtest[j, int(float(split_attr))] <= float(split_val):
                    jump = int(float(self.tree[i, 2]))
                else:
                    jump = int(float(self.tree[i, 3]))

                i += jump
                if i >= len(self.tree):
                    break

                split_attr = self.tree[i, 0]
                split_val = self.tree[i, 1]

        return y_pred

