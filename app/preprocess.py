import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RetailPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        
        X['Epidemic_Category'] = X['Category'] + '_' + X['Epidemic'].astype(str)
        X['Season_Category'] = X['Seasonality'] + '_' + X['Category']
        X['Discount_Capped'] = X['Discount'].clip(upper=15)
        X['has_high_discount'] = (X['Discount'] > 15).astype(int)
        X['Order_Fufillment_Rate'] = np.where(
            X['Units Ordered'] > 0,
            X['Units Sold'] / X['Units Ordered'],
            0
        )
        X = X.drop(columns=['Date', 'Store ID', 'Product ID', 'Discount'], axis=1)
        return X
