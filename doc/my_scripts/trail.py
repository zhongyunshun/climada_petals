# Extracting data for plotting
data = []
for year in years:
    for model, metrics in results[year].items():
        data.append({'year': year, 'model': model, 'train_roc_auc': metrics['train_roc_auc'], 'test_roc_auc': metrics['test_roc_auc']})

# Creating dataframe
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 6))

# Plot for LogisticRegression
plt.subplot(1, 2, 1)
sns.lineplot(data=df[df['model'] == 'LogisticRegression'], x='year', y='train_roc_auc', label='Train ROC AUC')
sns.lineplot(data=df[df['model'] == 'LogisticRegression'], x='year', y='test_roc_auc', label='Test ROC AUC')
plt.title('Logistic Regression ROC AUC over Years')
plt.xlabel('Year')
plt.ylabel('ROC AUC')
plt.legend()

# Plot for XGBClassifier
plt.subplot(1, 2, 2)
sns.lineplot(data=df[df['model'] == 'XGBClassifier'], x='year', y='train_roc_auc', label='Train ROC AUC')
sns.lineplot(data=df[df['model'] == 'XGBClassifier'], x='year', y='test_roc_auc', label='Test ROC AUC')
plt.title('XGBClassifier ROC AUC over Years')
plt.xlabel('Year')
plt.ylabel('ROC AUC')
plt.legend()

plt.tight_layout()
plt.show()