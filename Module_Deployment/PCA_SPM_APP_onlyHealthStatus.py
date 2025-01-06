import numpy as np
import pickle
import scipy
from matplotlib import pyplot as plt
from flask import Flask

app = Flask(__name__)

def computeMetricsContributions(PCAmodelData, processData):
    processData_scaled = PCAmodelData["scaler"].transform(processData)
    
    # compute scores and reconstruct
    score = PCAmodelData["PCAmodel"].transform(processData_scaled)
    score_reduced = score[:,0:PCAmodelData["n_comp"]]
    processData_scaled_reconstruct = np.dot(score_reduced, PCAmodelData["P_matrix"].T)
    
    # compute Q metric
    error = processData_scaled_reconstruct - processData_scaled
    Q = np.sum(error*error, axis = 1)[0] # [0] used to convert 1D array of size 1 into scalar
    
    # compute T2 metric
    T2 = np.dot(np.dot(score_reduced[0,:], PCAmodelData["lambda_k_inv"]), score_reduced[0,:].T)
    
    # Q contributions of variables
    Q_contri = error[0]*error[0] # vector of contributions
    
    # T2 contributions
    D = np.dot(np.dot(PCAmodelData["P_matrix"],PCAmodelData["lambda_k_inv"]),PCAmodelData["P_matrix"].T)
    T2_contri = np.dot(scipy.linalg.sqrtm(D),processData_scaled[0])**2 # vector of contributions
    
    return [Q, T2, Q_contri, T2_contri]

def getLatestProcessData():
    # fetch dummy process data from a local file
    processDataAll = np.loadtxt('processLatestDatabase_local.csv', delimiter=',')

    # pick a random row and return a 2D array with 1 row
    processData = processDataAll[np.random.randint(0,len(processDataAll)),:][None,:] 
    return processData

def runPCAmodel():
    # read saved PCA model data
    with open('PCAmodelData.pickle', 'rb') as f:
        PCAmodelData = pickle.load(f)
    
    # read latest process data
    processData = getLatestProcessData()
    
    # compute monitoring metrics and abnormality contributions
    [currentQ, currentT2, Q_contri, T2_contri] = computeMetricsContributions(PCAmodelData, processData)
    
    # detect fault
    if (currentQ > PCAmodelData["Q_CL"]) or (currentT2 > PCAmodelData["T2_CL"]):
        processSate = 'Issue Detected'
    else:
        processSate = 'All Good'
    
    return processSate

@app.route('/')
def assessProcessHealth():
    return runPCAmodel()

if __name__ == '__main__':
    app.run(debug=True)