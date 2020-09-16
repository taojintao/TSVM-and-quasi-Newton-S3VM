############################################################################################
# Application
# 	An Example of applications of SVM, TSVM and QN-S3VM in Mineral Prospectivity Mapping.
# 	We applied SVM, TSVM and QN-S3VM algorithms to example data and saved the outputs.
############################################################################################

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from TSVM import TSVM
from Quasi_Newton_S3VM import Quasi_Newton_S3VM


def svm_train():
	'''
	Train an optimal SVM with Example data

	'''
	data_all=pd.read_csv("Example data.csv")
	data_labeled=data_all[(data_all['sample']==1)|(data_all['sample']==-1)]
	print(data_labeled.shape)
	# Divide dataset into feature set and label set
	mine_data = data_labeled.iloc[:,3:12]
	mine_target = data_labeled.iloc[:,12]
	# Divide dataset into testing dataset and training dataset
	mine_train,mine_test,mine_target_train,mine_target_test = train_test_split(mine_data,mine_target,test_size = 0.8,random_state = 1)

	# Tune different parameters
	c_range = np.arange(0.1,10.1,0.1)
	gamma_range = np.arange(0.1,2.1,0.1)
	cv_scores = []
	for c in c_range:
		for gam in gamma_range:
			svm_mine = SVC(C=c,kernel='rbf',gamma=gam)
			# Select K-fold cross-validation and scoring
			scores = cross_val_score(svm_mine,mine_train,mine_target_train,cv=3,scoring='accuracy')
			cv_scores.append([c,gam,scores.mean()])

	df_result=pd.DataFrame(cv_scores,columns=['c','gamma','accuracy'])
	df_result.to_csv("Results/SVM_train.csv",index=False,encoding='ANSI')


	# Draw a heatmap of optimization of the parameters
	index_names=["{:.2}".format(gamma) for gamma in gamma_range]
	column_names=["{:.2}".format(c) for c in c_range]
	values=np.empty((gamma_range.shape[0],c_range.shape[0]))

	for c in c_range:
		for gam in gamma_range:
			value=df_result[(df_result['c']==c)&(df_result['gamma']==gam)]['accuracy']
			x=index_names.index("{:.2}".format(gam))
			y=column_names.index("{:.2}".format(c))
			values[x][y]=value

	values_df=pd.DataFrame(values,index=index_names,columns=column_names)

	fig, ax = plt.subplots(figsize=(30,5))
	# Draw a heatmap with the numeric values in each cell
	sns_plot=sns.heatmap(values_df[:],ax=ax,cmap='jet',cbar_kws={'label':'Accuracy'},xticklabels=4,yticklabels=4)

	font = {'family': 'Arial',
		 'weight': 'normal',
		 'size': 18,
		 }
	ax.set_xlabel('C')
	ax.set_ylabel('Gamma')
	plt.savefig("Results/Optimization_parameters.jpg")
	#plt.show()
def svm_predict():
	'''
	Predict testing data and all data using trained optimal SVM

	'''
	data_all=pd.read_csv("Example data.csv")
	data_labeled=data_all[(data_all['sample']==1)|(data_all['sample']==-1)]
	print(data_labeled.shape)
	# Divide dataset into feature set and label set
	mine_data = data_labeled.iloc[:,3:12]
	mine_target = data_labeled.iloc[:,12]
	# Divide dataset into testing dataset and training dataset
	mine_train,mine_test,mine_target_train,mine_target_test = train_test_split(mine_data,mine_target,test_size = 0.8,random_state = 1)

	# Select optimal SVM
	optimal_svm = SVC(C=0.3,kernel='rbf',gamma=0.7)
	optimal_svm.fit(mine_train,mine_target_train)

	# Predict labels of testing dataset
	mine_target_pred = optimal_svm.predict(mine_test)
	result_test=np.c_[mine_target_test,mine_target_pred]
	pd.DataFrame(result_test,columns=["sample","prediction"]).to_csv("Results/SVM_test.csv",index=False,encoding='ANSI')

	# Predict labels of all data
	mine_data_all = data_all.iloc[:,3:12]
	mine_pred=optimal_svm.predict(mine_data_all)

	# Get column names and convert them to list
	columns_list = data_all.columns.values.tolist()
	columns_list.append('prediction')
	result_predict=np.c_[data_all.values,mine_pred]
	pd.DataFrame(result_predict,columns=columns_list).to_csv("Results/SVM_prediction.csv",index=False,encoding='ANSI')

def tsvm_predict():
	'''
	Predict testing data and all data using trained optimal TSVM

	'''
	data_all=pd.read_csv("Example data.csv")
	data_label=data_all[(data_all['sample']==1)|(data_all['sample']==-1)]
	print("The number of labeled samples in all data：",data_label.shape)
	data_unlabel=data_all[data_all['sample']==0]
	print("The number of unlabeled samples in all data：",data_unlabel.shape)
	# Divide dataset into feature set and label set
	mine_label_data = data_label.iloc[:,3:12]
	mine_label_target = data_label.iloc[:,12]
	# Divide dataset into testing dataset and training dataset
	mine_train,mine_test,mine_target_train,mine_target_test = train_test_split(mine_label_data,mine_label_target,test_size = 0.8,random_state = 1)
	# Regard testing data as unlabeled data
	mine_unlabel=mine_test
	print("The number of unlabeled samples used:",mine_unlabel.shape)

	# Select optimal TSVM
	tsvm=TSVM(X_l=mine_train, y=mine_target_train, X_u=mine_unlabel,C_l=1.0,C_u=0.001,kernel='rbf',C=0.3,gamma=0.7)
	tsvm.train()

	# Predict labels of testing dataset
	mine_target_pred = tsvm.predict(mine_test)
	result_test=np.c_[mine_target_test,mine_target_pred]
	pd.DataFrame(result_test,columns=["sample","prediction"]).to_csv("Results/TSVM_test.csv",index=False,encoding='ANSI')

	# Predict labels of all data
	mine_data_all = data_all.iloc[:,3:12]
	mine_pred=tsvm.predict(mine_data_all)
	# Get column names and convert them to list
	columns_list = data_all.columns.values.tolist()
	columns_list.append('prediction')
	result_predict=np.c_[data_all.values,mine_pred]
	pd.DataFrame(result_predict,columns=columns_list).to_csv("Results/TSVM_prediction.csv",index=False,encoding='ANSI')

def qn_s3vm_train():
	'''
	Train an optimal S3VM with Example data

	'''
	data_all=pd.read_csv("Example data.csv")
	data_label=data_all[(data_all['sample']==1)|(data_all['sample']==-1)]
	print("The number of labeled samples in all data：",data_label.shape)
	data_unlabel=data_all[data_all['sample']==0]
	print("The number of unlabeled samples in all data：",data_unlabel.shape)
	# Divide dataset into feature set and label set
	mine_label_data = data_label.iloc[:,3:12]
	mine_label_target = data_label.iloc[:,12]
	# Divide dataset into testing dataset and training dataset
	mine_train,mine_test,mine_target_train,mine_target_test = train_test_split(mine_label_data,mine_label_target,test_size = 0.8,random_state = 1)
	# Regard testing data as unlabeled data
	mine_unlabel=mine_test
	print("The number of unlabeled samples used:",mine_unlabel.shape)

	kf=KFold(n_splits=3)

	# Tune different parameters
	lam_values=[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]
	lamU_values=[0.001,0.01,0.1,1,10,100]
	gamma_range = np.arange(0.1,2.1,0.1)
	cv_scores = []
	for lam_value in lam_values:
		for lamU_value in lamU_values:
			for gamma in gamma_range:
				correctRate=[]
				for train_index,test_index in kf.split(mine_train,mine_target_train):
					x_train,x_test=mine_train.values[train_index],mine_train.values[test_index]
					y_train,y_test=mine_target_train.values[train_index],mine_target_train.values[test_index]
					qn_s3vm=Quasi_Newton_S3VM(X_l=x_train, y=y_train, X_u=x_test, lam=lam_value, lam_u=lamU_value, kernel="rbf",sigma=gamma)
					qn_s3vm.fit()
					mine_target_pred = qn_s3vm.get_predictions(x_test)
					cm = confusion_matrix(y_test, mine_target_pred)
					cRate = (cm[1,1] + cm[0,0]) / cm.sum()
					correctRate.append(cRate)
				averagePrecision = sum(correctRate)/len(correctRate)
				print(averagePrecision)
				cv_scores.append([lam_value,lamU_value,gamma,averagePrecision])
	df_result=pd.DataFrame(cv_scores,columns=['lam','lamU','gamma','accuracy'])
	df_result.to_csv("Results/QN-S3VM_train.csv",index=False,encoding='ANSI')

def qn_s3vm_predict():
	'''
	Predict testing data and all data using trained optimal S3VM

	'''
	data_all=pd.read_csv("Example data.csv")
	data_label=data_all[(data_all['sample']==1)|(data_all['sample']==-1)]
	print("The number of labeled samples in all data：",data_label.shape)
	data_unlabel=data_all[data_all['sample']==0]
	print("The number of unlabeled samples in all data：",data_unlabel.shape)
	# Divide dataset into feature set and label set
	mine_label_data = data_label.iloc[:,3:12]
	mine_label_target = data_label.iloc[:,12]
	# Divide dataset into testing dataset and training dataset
	mine_train,mine_test,mine_target_train,mine_target_test = train_test_split(mine_label_data,mine_label_target,test_size = 0.8,random_state = 1)
	# Regard testing data as unlabeled data
	mine_unlabel=mine_test
	print("The number of unlabeled samples used:",mine_unlabel.shape)

	# Select optimal S3VM
	qn_s3vm=Quasi_Newton_S3VM(X_l=mine_train, y=mine_target_train, X_u=mine_unlabel, lam=0.0001, lam_u=1, kernel="rbf",sigma=0.5)
	qn_s3vm.fit()

	# Predict labels of testing dataset
	mine_target_pred = qn_s3vm.get_predictions(mine_unlabel)
	result_test=np.c_[mine_target_test,mine_target_pred]
	pd.DataFrame(result_test,columns=["sample","prediction"]).to_csv("Results/QN-S3VM_test.csv",index=False,encoding='ANSI')

	# Predict labels of all data
	mine_data_all = data_all.iloc[:,3:12]
	mine_pred=qn_s3vm.get_predictions(mine_data_all)
	# Get column names and convert them to list
	columns_list = data_all.columns.values.tolist()
	columns_list.append('prediction')
	result_predict=np.c_[data_all.values,mine_pred]
	pd.DataFrame(result_predict,columns=columns_list).to_csv("Results/QN-S3VM_prediction.csv",index=False,encoding='ANSI')

if __name__=="__main__":
	print("main")
	#svm_train()
	#svm_predict()
	#tsvm_predict()
	#qn_s3vm_train()
	#qn_s3vm_predict()

