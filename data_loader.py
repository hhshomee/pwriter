import pandas as pd
from typing import List, Dict, Any, Union, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file: str, limit: int):
    try:
        df= pd.read_csv(file)
        if limit is not None and limit>0:
            df=df[:limit]
        logger.info(f"Working on {limit} rows")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")




