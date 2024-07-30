


# Extracting data for plotting
data = []
for year in years:
    for model, metrics in results[year].items():
        data.append({'year': year, 'model': model, 'train_roc_auc': metrics['train_roc_auc'], 'test_roc_auc': metrics['test_roc_auc']})

# Creating dataframe
df = pd.DataFrame(data)

'''ROC AUC over years for LogisticRegression and XGBClassifier'''

# Plotting with y-axis range set from 0.85 to 1
plt.figure(figsize=(18, 8))

# Plot for LogisticRegression
plt.subplot(1, 2, 1)
sns.lineplot(data=df[df['model'] == 'LogisticRegression'], x='year', y='train_roc_auc', label='Train ROC AUC', marker='o')
sns.lineplot(data=df[df['model'] == 'LogisticRegression'], x='year', y='test_roc_auc', label='Test ROC AUC', marker='o')
for i in range(len(df[df['model'] == 'LogisticRegression'])):
    year = df[df['model'] == 'LogisticRegression'].iloc[i]['year']
    train_roc_auc = df[df['model'] == 'LogisticRegression'].iloc[i]['train_roc_auc']
    test_roc_auc = df[df['model'] == 'LogisticRegression'].iloc[i]['test_roc_auc']
    plt.text(year, train_roc_auc, f'{train_roc_auc:.2f}', fontsize=9, ha='right')
    plt.text(year, test_roc_auc, f'{test_roc_auc:.2f}', fontsize=9, ha='right')
plt.title('Logistic Regression ROC AUC over Years')
plt.xlabel('Year')
plt.ylabel('ROC AUC')
plt.ylim(0.85, 1)
plt.legend()
plt.xticks(years, rotation=45)
plt.grid(True)

# Plot for XGBClassifier
plt.subplot(1, 2, 2)
sns.lineplot(data=df[df['model'] == 'XGBClassifier'], x='year', y='train_roc_auc', label='Train ROC AUC', marker='o')
sns.lineplot(data=df[df['model'] == 'XGBClassifier'], x='year', y='test_roc_auc', label='Test ROC AUC', marker='o')
for i in range(len(df[df['model'] == 'XGBClassifier'])):
    year = df[df['model'] == 'XGBClassifier'].iloc[i]['year']
    train_roc_auc = df[df['model'] == 'XGBClassifier'].iloc[i]['train_roc_auc']
    test_roc_auc = df[df['model'] == 'XGBClassifier'].iloc[i]['test_roc_auc']
    plt.text(year, train_roc_auc, f'{train_roc_auc:.2f}', fontsize=9, ha='right')
    plt.text(year, test_roc_auc, f'{test_roc_auc:.2f}', fontsize=9, ha='right')
plt.title('XGBClassifier ROC AUC over Years')
plt.xlabel('Year')
plt.ylabel('ROC AUC')
plt.ylim(0.85, 1)
plt.legend()
plt.xticks(years, rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

'''Confusion Matrix'''

import numpy as np

# Summing confusion matrices for each model
logistic_conf_matrix = np.zeros((2, 2))
xgb_conf_matrix = np.zeros((2, 2))

for year in years:
    logistic_conf_matrix += np.array(results[year]['LogisticRegression']['confusion_matrix'])
    xgb_conf_matrix += np.array(results[year]['XGBClassifier']['confusion_matrix'])

# Plotting the confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Logistic Regression Confusion Matrix
sns.heatmap(logistic_conf_matrix, annot=True, fmt='g', ax=axes[0], cmap='Blues')
axes[0].set_title('Logistic Regression Total Confusion Matrix (2001-2023)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Negative', 'Positive'])
axes[0].set_yticklabels(['Negative', 'Positive'])

# XGBClassifier Confusion Matrix
sns.heatmap(xgb_conf_matrix, annot=True, fmt='g', ax=axes[1], cmap='Blues')
axes[1].set_title('XGBClassifier Total Confusion Matrix (2001-2023)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['Negative', 'Positive'])
axes[1].set_yticklabels(['Negative', 'Positive'])

plt.tight_layout()
plt.show()

'''Feature Importance'''
# Aggregating feature importances for each model
logistic_feature_importance = pd.DataFrame(columns=['feature', 'importance'])
xgb_feature_importance = pd.DataFrame(columns=['feature', 'importance'])

for year in years:
    logistic_feature_importance = logistic_feature_importance.append(results[year]['LogisticRegression']['feature_importance'], ignore_index=True)
    xgb_feature_importance = xgb_feature_importance.append(results[year]['XGBClassifier']['feature_importance'], ignore_index=True)

# Averaging feature importances
avg_logistic_feature_importance = logistic_feature_importance.groupby('feature').mean().reset_index()
avg_xgb_feature_importance = xgb_feature_importance.groupby('feature').mean().reset_index()

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Logistic Regression Feature Importance
sns.barplot(data=avg_logistic_feature_importance, x='importance', y='feature', ax=axes[0], palette='Blues_d')
axes[0].set_title('Logistic Regression Average Feature Importance (2001-2023)')
axes[0].set_xlabel('Importance')
axes[0].set_ylabel('Feature')

# XGBClassifier Feature Importance
sns.barplot(data=avg_xgb_feature_importance, x='importance', y='feature', ax=axes[1], palette='Blues_d')
axes[1].set_title('XGBClassifier Average Feature Importance (2001-2023)')
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('Feature')

plt.tight_layout()
plt.show()
