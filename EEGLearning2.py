# Classification Project: Use Simplified EEG data to predict weather or not
#subject understands what is happening

# Load libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Load dataset
url = '9.2-10.csv'
names = ["Subject","Location" ,"0.2" ,"0.4" ,"0.6" ,"0.8" ,"1","Condition"]
dataset = read_csv(url, header=None, names=names)
#change to int
for name in names:
    dataset[name] = dataset[name].astype("float64")


###descriptive stats
###shape
###type of data for each columne
##set_option('display.max_rows', 500)
##print(dataset.dtypes)
###print "header" (fist 20 rows)
##set_option('display.width', 100)
##print(dataset.head(20)) #we can see that we may have to normalize
####descriptions, change precision to 3 places
##set_option('precision', 3)
##print(dataset.describe())
### class distribution
##print(dataset.groupby('Condition').size())
##
### Data visualizations
##
### histograms
##dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
##pyplot.show()
### density
##dataset.plot(kind='density', subplots=True, layout=(10, 7), sharex=False, legend=False, fontsize=1)
##pyplot.show()
##
### scatter plot matrix
##print("hi")
##scatter_matrix(dataset)
##pyplot.show()
### correlation matrix
##fig = pyplot.figure()
##ax = fig.add_subplot(111)
##cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
##fig.colorbar(cax)
##pyplot.show()

##df = dataset 3(data, names=names)
##dataset.corr().to_csv("test.csv")
##sn.heatmap(corrMatrix, annot=True)
##plt.show()
# Prepare Data

# Split-out validation dataset
array = dataset.values
X = array[:,0:7].astype(float)
Y = array[:,7]

##model= ExtraTreesClassifier(n_estimators=100)
##model.fit(X, Y)
##print(model.feature_importances_)
#print(array)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Evaluate Algorithms

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'
##
##### Spot Check Algorithms
####models = []
####models.append(('LR', LogisticRegression(solver='liblinear')))
####models.append(('LDA', LinearDiscriminantAnalysis()))
####models.append(('KNN', KNeighborsClassifier()))
####models.append(('CART', DecisionTreeClassifier()))
####models.append(('NB', GaussianNB()))
####models.append(('SVM', SVC(gamma='auto')))
####results = []
####names = []
####
####for name, model in models:
####	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
####	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
####	results.append(cv_results)
####	
####	names.append(name)
####	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
######	print(msg)
######	model.fit(X_train, Y_train)
######	predictions = model.predict(X_validation)
######	print(predictions)
######	print(confusion_matrix(Y_validation, predictions))
####
##### Compare Algorithms
####
####fig = pyplot.figure()
####fig.suptitle('Algorithm Comparison')
####ax = fig.add_subplot(111)
####pyplot.boxplot(results)
####ax.set_xticklabels(names)
####pyplot.show()
##
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='auto'))])))
results = []
names = []
print(Y_validation)
for name, model in pipelines:
        
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	print(confusion_matrix(Y_validation, predictions))
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(n_estimators=2)))
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=2)))
results = []
names = []
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


