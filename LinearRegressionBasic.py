from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv("student_perf.csv")

print(df.isnull().sum())

type_dct = {str(k): list(v) for k, v in df.groupby(df.dtypes, axis=1)}
print(f"type_dict: {type_dct}")

type_counts = {}
for x in type_dct:
    type_counts[f"{x}"] = len(type_dct[x])

print(f"type_counts: {type_counts}")


labels = []
values = []
for x in type_counts:
    labels.append(x)
    values.append(type_counts[x])

plt.pie(values, labels = labels)



class MyOneHotEncoder:
    def __init__(self, df):
        self.col_unique_dict = {}
        self.df = df
        self.cols_encode = self.df.select_dtypes(include=['object']).columns.tolist() #categorical columns
        self.binary_cols = {} #for handling binary case
        
        
    def fit(self):
        for col in self.cols_encode:
            unique_vals = self.df[f"{col}"].unique()
            self.col_unique_dict[f"{col}"] =  unique_vals #categorical column: [list of unique values]
            if (len(unique_vals) == 2):
                self.binary_cols[col] = unique_vals[0]
            

    def transform(self):
        df_transform = self.df.copy()

        for col in self.cols_encode:
            if (col in self.binary_cols):
                ref_val = self.binary_cols[col]
                new_name = f"{ref_val}_{col}"
                df_transform[new_name] = (df[col] == ref_val).astype(int)

            else:
                for val in self.col_unique_dict[col]:
                    new_col_name = f"{col}_{val}"
                    df_transformed[new_col_name] = (df[col] == val).astype(int)
                df_transformed.drop(col, axis=1, inplace=True)

        return df_transform

    def fit_transform(self):
        self.fit()
        return self.transform()


encoder = MyOneHotEncoder(df)

new_df = encoder.fit_transform()







def compute_mse(y_true, y_pred):
    if (y_true.shape == y_pred.shape):
        squared_errors = np.square(y_true-y_pred)
        return np.mean(squared_errors)
    else:
        print("y_true and y_pred have varying size")



class MyStandardScaler: #using z score
    def __init__(self, df):
        self.means_ = {}
        self.stds_ = {}
        self.df = df
        self.cols_encode = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    def fit(self):
        for col in self.cols_encode:
            self.means_[col] = self.df[col].mean()
            self.stds_[col] = self.df[col].std()

    def transform(self):
        for col in self.cols_encode:
            self.df[col] = (self.df[col] - self.means_[col]) / self.stds_[col]
        return self.df
    
    
    
    
    
    
    
    
    
    
class LinearRegression:
    def __init__(self, X, y_true, alpha, num_epochs):
        self.learning_rate = alpha
        self.track_loss = []
        self.y_true = y_true
        self.m = len(self.y_true)
        self.X = np.hstack([np.ones((self.m, 1)), X]) #horizonatlly stack ones with X
        self.weights = np.zeros((self.X.shape[1], 1)) #initial weight vector is just zeros #missed the shape tuple
        self.num_epochs = num_epochs

    def calculate_weights(self):
        #runs gradient descent, returns weight vector
        for i in range(0,self.num_epochs):
            self.weights = self.weights - (((self.learning_rate/self.m) * self.X.T) @ ((self.X @ self.weights) - self.y_true))
            y_pred = self.find_y_pred()
            self.track_loss.append(compute_mse(self.y_true,y_pred))
        return self.weights

    def find_y_pred(self):
        y_pred = self.X @ self.weights #@ is matrix multiplication not * for elementwise
        return y_pred

    def plot_loss(self):
        epochs = list(range(0,self.num_epochs))
        plt.plot(epochs, self.track_loss) #x then y
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.show()
        
        
        
df.drop("Extracurricular Activities", axis=1, inplace = True)



correlations = df.corr()['Performance Index'].drop('Performance Index').abs().sort_values(ascending=False)

correlations.plot(kind='bar', color='skyblue')
plt.title("Absolute Correlation with Target")
plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")
plt.xticks(rotation=45)



plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


best_feature = correlations.idxmax()
print(f"Best feature: {best_feature} (Correlation = {correlations.max():.2f})") #best feature is previous scores just so anyone reading this knows



print(X_train.shape, type(X_train), X_train.to_numpy().reshape(-1, 1).shape)
print(y_train.shape)

model = LinearRegression(X_train.to_numpy().reshape(-1, 1), y_train.to_numpy().reshape(-1, 1), alpha = 0.01, num_epochs = 1000)
weights = model.calculate_weights()
model.plot_loss()



print(f"weights: {model.weights[0][0]}") #w0 or bias term
print(f"bias: {model.weights[1][0]}") #w1, weight vector
print(f"\n\n\nweights vector: {model.weights}")






