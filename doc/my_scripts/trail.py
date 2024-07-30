
# Plotting with all years on x-axis and markers with ROC AUC values
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
plt.legend()
plt.xticks(years, rotation=45)

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
plt.legend()
plt.xticks(years, rotation=45)

plt.tight_layout()
plt.show()