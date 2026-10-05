import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

def multilabel_metrics(y_true,y_prob,threshold=0.5):
    y_true=np.asarray(y_true); y_prob=np.asarray(y_prob); y_pred=(y_prob>=threshold).astype(int)
    auc={}; ap={}
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:,i]))>1:
            auc[i]=float(roc_auc_score(y_true[:,i],y_prob[:,i]))
            ap[i]=float(average_precision_score(y_true[:,i],y_prob[:,i]))
    return {"macro_auroc":float(np.mean(list(auc.values()))) if auc else float("nan"),
            "macro_auprc":float(np.mean(list(ap.values()))) if ap else float("nan"),
            "micro_f1":float(f1_score(y_true,y_pred,average="micro",zero_division=0)),
            "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
            "precision":float(precision_score(y_true,y_pred,average="micro",zero_division=0)),
            "recall":float(recall_score(y_true,y_pred,average="micro",zero_division=0)),
            "per_class_auroc":auc,"per_class_auprc":ap}

def selective_risk(y_true,y_prob,uncertainty,coverage=0.8,threshold=0.5):
    order=np.argsort(uncertainty); n=max(1,int(len(order)*coverage)); keep=order[:n]
    pred=(y_prob[keep]>=threshold).astype(int); err=np.abs(pred-y_true[keep]).mean()
    return {"coverage":coverage,"selective_risk":float(err),"n":int(n)}
