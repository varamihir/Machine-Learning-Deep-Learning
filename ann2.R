# Artificial Neural Network(ANN)
# Classification Template that we used in logistic regression model
# Import the data set
dataset <- read.csv("Churn_Modelling.csv")
dataset <- dataset[,4:14]
str(dataset)
# lets convert Geography and gender levels into labels 1,2,3 for easy calculations and then into numeric
dataset$Geography <- factor(dataset$Geography,
                                       levels = c("France", "Germany","Spain"),
                                     labels = c(1,2,3))
dataset$Geography <- as.numeric(dataset$Geography)
dataset$Gender <- factor(dataset$Gender,
                                    levels = c("Female", "Male"),
                                    labels = c(1,2))
dataset$Gender <- as.numeric(dataset$Gender)
str(dataset)
# Spilitting  dataset into training and testing data sets
library(caTools)
set.seed(1234)
split <- sample.split(dataset$Exited, SplitRatio = 0.8) 
training_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split== FALSE)

# Feature scaling is needed for ANN. Training in ANN is highly compute parallel intensive calculations and besides it is required by package
training_set[-11] <- scale(training_set[-11])
testing_set[-11]  <- scale(testing_set[-11])                         

# fitting ANN to the training set
#install.packages("h20")
library(h2o)
h2o.init(nthreads = -1)
classifier <- h2o.deeplearning(y = "Exited",
                               training_frame = as.h2o(training_set),
                               activation = "Rectifier",
                               hidden = c(6,6),
                               epochs = 100,
                              train_samples_per_iteration = -2)
# prediction on the test set
prob_pred <- h2o.predict(classifier, newdata = as.h2o(testing_set[-11]))
# # Creating a vector of predicted results in 0 and 1 and it is easy to interpret the results
y_pred <- (prob_pred > 0.5)# this is still in h2o environment
# covert into vector from h2o environment
y_pred <- as.vector(y_pred)
y_pred
cm <- table(testing_set[,11],y_pred)
cm
# accuracy is 85%

h2o.shutdown()# disconnected from h2o server
Y

