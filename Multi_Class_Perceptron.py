import numpy as np
import pandas as pd


########################################################################################################
#                                  P R E - P R O C E S S I N G
##########################################################################################################
# Loading Train and Test Datsets 
Train_dataset = pd.read_csv("D:\Liverpool Programming\Data Mining\CA1data/train.data",header= None)
Test_dataset = pd.read_csv("D:\Liverpool Programming\Data Mining\CA1data/test.data",header= None)

# Masking Out class1 , class 2 , class 3 from Training Set 
mask = (Train_dataset.values[:,4] == 'class-1')
Train_class_1 = Train_dataset.values[mask,:]

mask = (Train_dataset.values[:,4] == 'class-2')
Train_class_2 = Train_dataset.values[mask,:]

mask = (Train_dataset.values[:,4] == 'class-3')
Train_class_3 = Train_dataset.values[mask,:]


# Masking Out class1 , class 2 , class 3 from Test Set 
mask = (Test_dataset.values[:,4] == 'class-1')
Test_class_1 = Test_dataset.values[mask,:]

mask = (Test_dataset.values[:,4] == 'class-2')
Test_class_2 = Test_dataset.values[mask,:]

mask = (Test_dataset.values[:,4] == 'class-3')
Test_class_3 = Test_dataset.values[mask,:]

#################################################################
#                 P E R C E P T R O N 
#################################################################

class Multi_Class_Perceptron():
    def __init__(self, Iter, All_data , Data_1):
        """ 
        This is initializer method which loads the data on which the model has to be Trained.
        
        Iter: No of Iterations
        All Data: All Data
        Data1: CLass of Data to be classified
        """ 
        # Loading Data as self 
        self.iter= Iter 
        self.data1= Data_1
        self.data2= All_data

        #Creating a combined dataset and shuffling them for better training.
        Dataset= np.array(self.data2)
        np.random.shuffle(Dataset)
        print("Randomized Training Data = " , Dataset)
        
        # Creating y (actual classes), from dataset
        classes= Dataset[:,[4]]
        
        # Converting Classes in the form of +1 and -1 (eg: class1 = 1 ; class2 = -1 )
        for i in range(len(classes)):
            if classes[i] ==  self.data1[0][4]:
                classes[i]= 1
            else:
                classes[i]= -1
        
        # Final Training features (X_Train) and Labelled Training Class (Y_Train) 
        self.X_Train= Dataset[:,[0,1,2,3]]      
        self.Y_Train= classes
        
 # Train Function
    def train(self):

        # Initializing Weights and Bias 
        self.weights= np.zeros(shape=[1,(len(self.X_Train[0]))])
        self.bias=0

        # For number of Iterations and every row in X_Train calculating Activation Score 
        for i in range(self.iter):
            for j in range(len(self.X_Train)):
                
                a= np.dot(self.X_Train[j] , np.transpose(self.weights)) + self.bias  # Activation score 
                
                # if activation score multiplied by the actual class from Y_Train is negative that means wrong classifiication  
                if self.Y_Train[j]*a <= 0:

                    self.weights = self.weights + self.Y_Train[j]*self.X_Train[j]  # New Updated Weights  
                    
                    self.bias= self.bias + self.Y_Train[j]     # New Updated Weights
    
        # return final weights and bias  
        return self.weights , self.bias



    def test(self, All_data, TestData1 ,weights, bias):
        """
        This function is used to test the model (Calculated weights and biases)
        
        All_data: All Data To test 
        TestData1: Second Data to test 
        Weights: the weights calculated from train
        bias: The bias calculated from train 
        """
        # Counter for number of right prediction 
        right = 0

        #Creating a combined dataset and shuffling them for better training.
        Test_Data= np.array(All_data)
        print("Test Data : ", Test_Data)
        
        # Checking activation score of each input of test data

        for i in range(len(Test_Data)):
            a = np.dot(Test_Data[i][0:4] , np.transpose(weights)) + bias
            
            # If a > 0 and the Test Data classified class is equal to actual class -> Right Prediction
            if a > 0:

                if  Test_Data[i,[4]][0]== TestData1[1,[4]][0]:
                    print("Activation Score = ", round(a[0],2) ,"Classified as ",TestData1[1,[4]][0] ,"correct classification")
                    right+=1
                else:
                    print("Activation Score = ", round(a[0],2) ,"Classified as ", TestData1[1,[4]][0] ,"Wrong classification")

            # If a > 0 and the Test Data classified class is equal to actual class -> Right Prediction    
            elif a  < 0:



                if  Test_Data[i,[4]][0] != TestData1[1,[4]][0]:
                    print("Activation Score = ", round(a[0],2) ,"Classified as NOT ", TestData1[1,[4]][0] ,"correct classification")
                    right+=1
                else:
                    print("Activation Score = ", round(a[0],2) ,"Classified as NOT ", TestData1[1,[4]][0] ,"Wrong classification")

        # printing Accuracy  and Plotting 
        print("\n Out of ",len(Test_Data),right, " classifications are right ")
        print("Accuracy = ", right/(len(Test_Data)))
        

#################################################################################################
#                               Running Code
#################################################################################################
p1= Multi_Class_Perceptron(20,Train_dataset, Train_class_1)
weights , bias = p1.train() 
print(weights, bias)
p1.test(Test_dataset, Test_class_1,weights,bias)


