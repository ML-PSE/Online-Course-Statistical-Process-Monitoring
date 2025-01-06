import numpy as np
import pickle
import scipy
from matplotlib import pyplot as plt
from flask import Flask, render_template

app = Flask(__name__, template_folder='.', static_folder='static')

def generateMetricPlots(currentQ, currentT2, Q_CL, T2_CL):
    # load historical metrics
    try:
        with open('PCAmetrics_history.pickle', 'rb') as f:
            PCAmetrics_history = pickle.load(f) # PCAmetrics_History = [Q_history, T2_history]; saving list instead of dictionary for illustration
    except:
        Q_history = []
        T2_history = []
        PCAmetrics_history = [Q_history, T2_history]
        
    # update metric history and save
    PCAmetrics_history[0].append(currentQ)
    PCAmetrics_history[1].append(currentT2)
    
    with open('PCAmetrics_history.pickle', 'wb') as f:
        pickle.dump(PCAmetrics_history, f, pickle.HIGHEST_PROTOCOL)
    
    # generate plot and save
    plt.figure(figsize=[15,4])
    plt.subplot(1,2,1)
    plt.plot(PCAmetrics_history[0], marker='o', label='value')
    plt.plot([0,len(PCAmetrics_history[0])],[Q_CL,Q_CL], color='red', label='threshold', linestyle='dashed')
    plt.xlabel('Sample # (rightmost data is latest))'), plt.ylabel('Q metric')
    
    plt.subplot(1,2,2)
    plt.plot(PCAmetrics_history[1], marker='o', label='value')
    plt.plot([0,len(PCAmetrics_history[1])],[T2_CL,T2_CL], color='red', label='threshold', linestyle='dashed')
    plt.xlabel('Sample # (rightmost data is latest)'), plt.ylabel('T$^2$ metric')
    
    plt.suptitle('Process Monitoring Metrics')
    plt.savefig('static/metricPlot.png')

def generateContributionPlots(Q_contri, T2_contri):
    # generate plot and save
    plt.figure(figsize=[15,4])
    plt.subplot(1,2,1)
    plt.bar(['variable ' + str((i+1)) for i in range(len(Q_contri))], Q_contri)
    plt.ylabel('Q contributions')
    plt.xticks(rotation = 80)
    
    plt.subplot(1,2,2)
    plt.bar(['variable ' + str((i+1)) for i in range(len(T2_contri))], T2_contri)
    plt.ylabel('T$^2$ contributions')
    plt.xticks(rotation = 80)
    
    plt.suptitle('Contribution Plots for Fault Diagnosis')
    plt.tight_layout()
    plt.savefig('static/contributionPlot.png')

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
    
    # generate and save metric plot containing historical metrics and current metrics
    generateMetricPlots(currentQ, currentT2, PCAmodelData["Q_CL"], PCAmodelData["T2_CL"]) # saves 'metricPlot.png' in the working folder
    
    # generate and save contribution plots for current sample
    generateContributionPlots(Q_contri, T2_contri) # saves 'contributionPlot.png' in the working folder
    
    return processSate

@app.route('/')
def assessProcessHealth():
    processState = runPCAmodel()

    # Set color based on process state
    text_color = "green" if processState == "All Good" else "red"

    # create appropriate HTML content
    # html_output = render_template('html_template.html', process_state=processState, text_color=text_color)
    html_output = render_template('html_template_autoRefresh.html', process_state=processState, text_color=text_color)

    return html_output

if __name__ == '__main__':
    app.run(debug=True)