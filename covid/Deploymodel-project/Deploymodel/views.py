from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request,'home.html')

def result(request):

    clf  = joblib.load("final_model.sav")
    lis = []
    lis.append(request.GET['Age_above_65'])
    lis.append(request.GET['Age_percintil'])
    lis.append(request.GET['gender'])
    lis.append(request.GET['Bp_diastolic_mean'])
    lis.append(request.GET['Bp_sistolic_mean'])
    lis.append(request.GET['Bp_diastolic_median'])
    lis.append(request.GET['Bp_sistolic_median'])
    lis.append(request.GET['Bp_diastolic_min'])
    lis.append(request.GET['Bp_sistolic_min'])
    lis.append(request.GET['Bp_diastolic_max'])
    lis.append(request.GET['Bp_sistolic_max'])
    lis.append(request.GET['Bp_diastolic_diff'])
    lis.append(request.GET['Bp_sistolic_diff'])
    lis.append(request.GET['Bp_diastolic_diff_rel'])
    lis.append(request.GET['Bp_sistolic_diff_rel'])

    ans = clf.predict([lis])
    
    return render(request,'result.html',{'ans': ans})