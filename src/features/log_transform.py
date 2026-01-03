from typing import List 

import pandas as pd
import numpy as np

def log_transform(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    result = df.copy()

    for feature in features:
        if feature in result.columns:
            result[feature] = np.log1p(result[feature])
    
    return result



                  
                
    
    
   