# #!./.conda/envs/py310/bin/python3.10
# Multi-task model
import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt

from sklearn.metrics import fbeta_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, log_loss, mean_absolute_error, \
    precision_recall_curve

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import warnings
from xgboost import DMatrix
import time
import pickle
import colorsys
import random
from proj.PSM import random_light_color
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


#
# if torch.cuda.is_available():
#     x_train = x_train.to(device)
#     x_eval = x_eval.to(device)
#     Y_train = Y_train.to(device)
#     y_eval = y_eval.to(device)
#     X_test = X_test.to(device)
#     y_test = y_test.to(device)


# clf = MultiOutputClassifier(XGBClassifier(learning_rate = 0.1))
# clf = clf.to(device)
# clf2 = MultiOutputClassifier(DecisionTreeClassifier())
# pipeline = Pipeline([('clf',clf)])
# model = pipeline.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# Print classification report on test data
# print('\n',classification_report(y_test.values, y_pred, target_names=y_test.columns.values))
# print("The F1 Micro Score is: {0:.2f}".format(f1_score(y_pred, y_test, average='micro')))
# print("The F1 Macro Score (Unweighted average) is: {0:.2f}".format(f1_score(y_pred, y_test, average='macro')))

class XGB_clf():
    def __init__(self, if_save=False):
        self.model_name = 'xgb'
        self.saveif = if_save
        # self.single_model = self.bi_clf()

    def random_dark_color(self):
        h = random.random()
        s = random.uniform(0.5, 0.8)
        v = random.uniform(0.2, 0.4)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return r, g, b

    # def F1_beta_score(self, preds, dtrain, beta=1.0):
    #     """
    #     MultiOutput F1_beta_score
    #
    #     This is a custom metric which is created with f1_beta.
    #
    #     It is used in gridsearch for scoring.
    #
    #     Since recall is very important to not miss important help issues, we choice beta as 2.
    #     Check details for beta: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
    #
    #     """
    #     # labels = dtrain.get_label()
    #     score_sum = 0
    #     for column in range(0, dtrain.shape[1]):
    #         score = fbeta_score(dtrain, preds, beta=beta, average='binary')
    #         # if y_true.columns[column] in set(critical_types):
    #         #     score = score * 4
    #         score_sum += score
    #     avg_f1_beta = score_sum / dtrain.shape[1]
    #     return avg_f1_beta

    # def F1_(self,preds,dtrain):
    #     # labels = dtrain.get_label()
    #     return 'F1_',f1_score(labels, preds, average='binary')

    def bi_clf(self, data, tag, savePath, lr=0.1, max_depth=3, min_cw=6, min_sl=30, rg=0.1, beta=1.0, use_beta=False):
        self.tag = tag
        if type(tag)==type([1]) and len(tag)==1:
            data = data[~data['embedding'].isna()].reset_index(drop=True)
            y = data[self.tag]
            X = data['embedding']


            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            self.x_train, self.x_eval, self.Y_train, self.Y_eval = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            dtrain = DMatrix(self.x_train, label=self.Y_train)
            dval = DMatrix(self.x_eval, label=self.Y_eval)

            # eval_set = [(dtrain, 'train'), (dval, 'eval')]
            eval_set = [(self.x_train, self.Y_train), (self.x_eval, self.Y_eval)]
            clf = XGBClassifier(use_label_encoder=False, object='binary:hinge', booster='gbtree', max_depth=max_depth,
                            min_sample_leaf=min_sl, min_child_weight=min_cw, learning_rate=lr, reg_lambda=rg)
            print("Start training")

            clf.fit(X=self.x_train, y=self.Y_train,
                    eval_metric=['logloss', 'error'],
                    early_stopping_rounds=10,
                    eval_set=eval_set,
                    verbose=False)
            self.evals_result = clf.evals_result()
            self.single_model = clf

        elif type(tag)==type([1]) and len(tag)>1:
            X = pd.DataFrame(data['embedding'].tolist())
            y = data[self.tag]
            y = y.astype('category')

            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            self.x_train, self.x_eval, self.Y_train, self.Y_eval = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            dtrain = DMatrix(self.x_train.values, label=self.Y_train.values, enable_categorical=True)
            dval = DMatrix(self.x_eval.values, label=self.Y_eval.values, enable_categorical=True)

            print("Start training")
            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'error'],
                'eta': lr,
                'max_depth': max_depth,
                "reg_lambda" : rg
            }

            # Lists to store evaluation results
            self.evals_result = {}

            # Training
            clf = xgb.train(params,
                            dtrain,
                            num_boost_round=1000,
                            evals=[(dtrain, 'validation_0'), (dval, 'validation_1')],
                            early_stopping_rounds=10,
                            evals_result=self.evals_result,
                            verbose_eval=10)
            self.single_model = clf

        # return clf
        if self.saveif:
            local = time.strftime('%Y%m%d%H%M%S', time.localtime())
            # pickle.dump(clf, open("pima.pickle.dat", "wb"))
            self.single_model.save_model(savePath + '/xgb_clf_%s.xgb' % local)

    def plot_result(self, savePath):
        train_logloss = self.evals_result['validation_0']['logloss']
        val_logloss = self.evals_result['validation_1']['logloss']
        train_error = self.evals_result['validation_0']['error']
        val_error = self.evals_result['validation_1']['error']

        if len(self.tag)==1:
            y_pred_train = self.single_model.predict_proba(self.x_train)[:, 1]
            y_pred_val = self.single_model.predict_proba(self.x_eval)[:, 1]
            # Compute ROC-AUC and plot
            roc_auc_train = roc_auc_score(self.Y_train.astype(int), y_pred_train)
            roc_auc_val = roc_auc_score(self.Y_eval.astype(int), y_pred_val)
            fpr_train, tpr_train, _ = roc_curve(self.Y_train.astype(int), y_pred_train)
            fpr_val, tpr_val, _ = roc_curve(self.Y_eval.astype(int), y_pred_val)

            precision, recall, thresholds = precision_recall_curve(self.Y_train.astype(int), y_pred_train)
            precision_, recall_, thresholds_ = precision_recall_curve(self.Y_eval.astype(int), y_pred_val)

            plt.figure(figsize=(24, 6))
            plt.subplot(1, 3, 1)
            plt.plot(train_logloss, 'b-', label='Train logLoss', linewidth=2)
            plt.plot(val_logloss, 'green', linestyle='-', label='Validation logLoss', linewidth=2)
            plt.plot(train_error, 'black', linestyle='-', label='Train error', linewidth=2)
            plt.plot(val_error, 'red', linestyle='-', label='Validation error', linewidth=2)

            # # Plot markers every 10 points
            # for i in range(0, len(train_logloss), 10):
            #     plt.plot(i, train_logloss[i], 'bo', markersize=8)
            #
            # for i in range(0, len(val_logloss), 10):
            #     plt.plot(i, val_logloss[i], 'green',marker='o',markersize=8)
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.xlabel('Boosting Iteration')
            plt.ylabel('Log Loss')
            plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
            plt.title('Training and Validation Loss Curves')

            # Plot ROC curve
            plt.subplot(1, 3, 2)
            plt.plot(fpr_train, tpr_train, 'b-', label=f'AUC: {roc_auc_train:.4f}', linewidth=2)
            plt.plot(fpr_val, tpr_val, 'green', linestyle='-', label=f'AUC: {roc_auc_val:.4f}', linewidth=2)
            plt.plot(fpr_val, fpr_val, 'r--', label='Random Classifier')
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

            plt.subplot(1, 3, 3)
            plt.plot(precision, recall, 'b-', label='Train PR curve', linewidth=2)
            plt.plot(precision_, recall_, 'green', linestyle='-', label='Validation PR curve', linewidth=2)
            # plt.plot(fpr_val, fpr_val, 'r--', label = 'Random Classifier')
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision/Recall Curve')
            plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

            plt.savefig(savePath + '/%s_分类模型训练过程.png' % self.tag.values)
            plt.close()
            print('成功保存')

        elif len(self.tag)>1:
            y_pred_train = self.single_model.predict(DMatrix(self.x_train))
            y_pred_train = np.array([[round(i) for i in t] for t in y_pred_train])
            y_pred_val = self.single_model.predict(DMatrix(self.x_eval))
            y_pred_val = np.array([[round(i) for i in t] for t in y_pred_val])
            # Compute ROC-AUC and plot
            roc_auc_train = roc_auc_score(self.Y_train.astype(int), y_pred_train)
            roc_auc_val = roc_auc_score(self.Y_eval.astype(int), y_pred_val)

            print('开始画图')
            plt.figure(figsize=(24, 6))
            plt.subplot(1, 3, 1)
            plt.plot(train_logloss, 'b-', label='Train logLoss', linewidth=2)
            plt.plot(val_logloss, 'green', linestyle='-', label='Validation logLoss', linewidth=2)
            plt.plot(train_error, 'black', linestyle='-', label='Train error', linewidth=2)
            plt.plot(val_error, 'red', linestyle='-', label='Validation error', linewidth=2)
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.xlabel('Boosting Iteration')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.title('Training and Validation Loss Curves')

            # Plot ROC curve
            plt.subplot(1, 3, 2)
            for t in range(0,len(self.tag)):
                fpr_train, tpr_train, _ = roc_curve(self.Y_train.astype(int).values[:,t], y_pred_train[:,t])
                fpr_val, tpr_val, _ = roc_curve(self.Y_eval.astype(int).values[:,t], y_pred_val[:,t])

                plt.subplot(1, 3, 2)
                plt.plot(fpr_train, tpr_train, self.random_dark_color(), linestyle='-', label=f'{self.tag[t]} Train AUC: {roc_auc_train:.4f}', linewidth=2)
                plt.plot(fpr_val, tpr_val, random_light_color(), linestyle='-', label=f'{self.tag[t]} Eval AUC: {roc_auc_val:.4f}', linewidth=2)
                plt.plot(fpr_val, fpr_val, 'r--', label='Random Classifier')
                if t==0:
                    plt.plot(fpr_val, fpr_val, 'r--', label='Random Classifier')
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.title('Receiver Operating Characteristic Curve')

            plt.subplot(1, 3, 3)
            # plt.plot(fpr_val, fpr_val, 'r--', label = 'Random Classifier')
            for t in range(0, len(self.tag)):
                precision, recall, thresholds = precision_recall_curve(self.Y_train.astype(int).values[:, t],
                                                                       y_pred_train[:, t])
                precision_, recall_, thresholds_ = precision_recall_curve(self.Y_eval.astype(int).values[:, t],
                                                                          y_pred_val[:, t])

                plt.subplot(1, 3, 3)
                plt.plot(precision, recall, self.random_dark_color(), linestyle='-',
                         label=f'{self.tag[t]} Train PR curve', linewidth=2)
                plt.plot(precision_, recall_, random_light_color(), linestyle='-', label=f'{self.tag[t]} Eval PR curve',
                         linewidth=2)
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision/Recall Curve')
            plt.legend()


            plt.savefig(savePath + '/%s_分类模型训练过程.png' % str(self.tag[1:-1]))
            print('成功保存')
            plt.close()

    def ala_result(self, savePath):
        predictions = self.single_model.predict(DMatrix(self.X_test))
        predictions = np.array([[round(i) for i in t] for t in predictions])
        # print('\n', classification_report(y_test.values, predictions, target_names=y_test.columns.values))
        for i in range(0,len(self.tag)):
            print(
                "--------------------------------------------- " + self.tag[i] + "--------------------------------------------- ")
            print(classification_report(self.y_test.astype(int).values[:,i], predictions[:,i]))
            print('\n')
            ## get the result score
            print("The F1 Micro Score is: {0:.2f}".format(f1_score(predictions[:,i], self.y_test.astype(int).values[:,i], average='micro')))
            print("The F1 Macro Score (Unweighted average) is: {0:.2f}".format(
                f1_score(predictions[:,i], self.y_test.astype(int).values[:,i], average='macro')))
            print("roc_auc %s: %.4f" % (self.tag[i], roc_auc_score(self.y_test.astype(int).values[:,i], predictions[:,i])))
        print('分析结束')

# params = {'clf__estimator__max_depth': [3,6,9],
#           "clf__estimator__min_samples_leaf": [10,30,50],
#         "clf__estimator__min_child_weight": [3,5],
#           "clf_reg_lambda": [0,0.1,0.5]
#           }
#
# from sklearn.metrics import make_scorer
# #
# # scorer = make_scorer(multiOutputF1_beta_score,beta=2)
#
# best_score = 0
# for max_dep in [3,6,9]:
#     for sample_leaf in [10,30,50]:
#         for child in [3,5]:
#             for L2 in [0,0.1,0.5]:
#                 for beta in [1.5,2]:
#                     a = time.time()
#                     clf = MultiOutputClassifier(XGBClassifier(max_depth = max_dep,min_sample_leaf = sample_leaf,
#                                                               min_child_weight = child, learning_rate = 0.1, reg_lambda = L2))
#                     clf = clf.to(device)
#                     print('Start training on params '+str(max_dep)+" "+str(sample_leaf)+" "+str(child)+" "+str(L2)+" "+str(beta))
#                     clf.fit(x_train,Y_train)
#                     predictions = clf.predict(x_eval)
#                     score = multiOutputF1_beta_score(y_eval, predictions, beta=beta)
#                     print("Training time: "+str(time.time()-a)+" s")
#                     if score > best_score:
#                         best_score = score
#                         best_params = {'max_depth': max_dep,
#                                         "min_samples_leaf": sample_leaf,
#                                         "min_child_weight": child,
#                                         "reg_lambda": L2,
#                                        "learning_rate":0.1
#                                        }
#                         best_beta = beta
#                         model_best = clf
# print("Selecting time: "+str(time.time()-a)+" s")
# print("Best score on validation set:{:.3f}".format(best_score))
# print("Best beta on validation set:{:.1f}".format(best_beta))
# print('Best parameters set:')
# for param_name in sorted(best_params.keys()):
#     print('\t%s: %r' % (param_name, best_params[param_name]))
#
# # 1.保存整个网络
# torch.save(model_best, 'data/model_best_xgb.pth')
# print(" Model saved")
#
# predictions = model_best.predict(X_test)
# # print('\n', classification_report(y_test.values, predictions, target_names=y_test.columns.values))
#
# for column in y_test.columns:
#     print("---------------------" + column + "--------------------------------------------- ")
#     print(classification_report(y_test.iloc[:,y_test.columns.get_loc(column)],predictions[:,y_test.columns.get_loc(column)]))
#
# ## get the result score
# print("The F1 Micro Score is: {0:.2f}".format(f1_score(predictions, y_test, average='micro')))
# print("The F1 Macro Score (Unweighted average) is: {0:.2f}".format(f1_score(predictions, y_test, average='macro')))
# print("The F1_Beta Score with Beta={0:.f} is:  {1:.2f}".format(best_beta,multiOutputF1_beta_score(y_test, predictions, beta=best_beta)))
#
# for i in range(0,len(Tag_list)):
#     auc = roc_auc_score(y_test.loc[:, y_test.columns[i]].values, predictions[:, i])
#     print("ROC AUC %s: %.4f" % (Tag_list[i],auc))
# 1.1加载参数
# model = torch.load('model.pth')
# model = MultiOutputClassifier(XGBClassifier(**best_params))
# model = model.to(device)
# if torch.cuda.is_available():
#     X_train = X_train.to(device)
#     y_train = y_train.to(device)


# grid_search = GridSearchCV(pipeline, params, n_jobs=-1, verbose=2,scoring=scorer)
# grid_search.fit(x_train, y_train)
# print('Best score: %0.3f' % grid_search.best_score_)
# print('Best parameters set:')
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(params.keys()):
#     print('\t%s: %r' % (param_name, best_parameters[param_name]))

# predictions = grid_search.predict(x_eval)

# Print classification report on training data
# print('\n', classification_report(y_eval.values, predictions, target_names=y_eval.columns.values))

# ## get the result score
# print("The F1 Micro Score is: {0:.2f}".format(f1_score(predictions, y_eval, average='micro')))
# print("The F1 Macro Score (Unweighted average) is: {0:.2f}".format(f1_score(predictions, y_test, average='macro')))
# print("The F1_Beta Score with Beta=2 is:  {0:.2f}".format(multiOutputF1_beta_score(y_test, predictions, beta=2)))
#
# for column in y_test.columns:
#     print("---------------------" + column + "--------------------------------------------- ")
#     print(classification_report(y_test.iloc[:,y_test.columns.get_loc(column)],predictions[:,y_test.columns.get_loc(column)]))


# Build Bert Model and get emebeddings
