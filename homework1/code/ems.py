'''
这部分代码是对于模型的评估
但可能因为时间上的不充裕
各部分代码耦合十分严重
'''
from typing import Dict
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os
output_dir = "../output/"
def output(X_test,y_test,result:Dict) -> None :
    for kernel,info in result.items():
        plt.figure()
        
        model = info['best_estimator']

        y_pred = model.predict(X_test)

        y_score = model.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print(f"{kernel} AUC = {roc_auc:.3f}")

        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx])/2
        print(f"{kernel} ERR = {eer:.3f}")

        # 保存 ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}, EER={eer:.3f}')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve - {kernel}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'ROC_{kernel}.png'))
        plt.close()

        # 混淆矩阵
        plt.figure()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix - {kernel}')
        plt.savefig(os.path.join(output_dir, f'CM_{kernel}.png'))
        plt.close()
                
