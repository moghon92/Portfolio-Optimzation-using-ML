""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
A  wrapper for BagLearner for regression.		  	   		   	 		  		  		    	 		 		   		 		  
  		   		   	 		  		  		    	 		 		   		 		  	  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
from scipy import stats
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class BagLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Bag Learner.
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs, bags, boost = False, verbose = False):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.bag_learners = []

        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
  		  	   		   	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "mghoneim3"  # replace tb34 with your Georgia Tech username

    def sample_bags(self, data_x, data_y):
        indices = np.random.randint(data_x.shape[0], size=data_x.shape[0])
        return data_x[indices, :], data_y[indices]

    def add_evidence(self, data_x, data_y):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		  

        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        for i in range(0, self.bags):
            l_i = self.learner(**self.kwargs) ## create instance i of the learner class
            sampled_data_x, sampled_data_y = self.sample_bags(data_x, data_y)
            l_i.add_evidence(sampled_data_x, sampled_data_y) ## add eveidence to this instance
            self.bag_learners.append(l_i) ## save this learner


    def query(self, xtest):
        """
        Estimate a set of test points given the model we built.
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        bag_preds = np.zeros(xtest.shape[0])
        for bl in self.bag_learners:  # query each learner
            y_pred_i = bl.query(xtest)  # predictions of learner_i
            bag_preds = np.vstack((bag_preds, y_pred_i))

        if self.bags >= 1:
            bag_preds = bag_preds[1:,:]

        y_pred = stats.mode(bag_preds, axis=0)[0][0]

        return y_pred

