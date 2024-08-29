import sys
import pandas as pd

from pathlib import Path
sys.path.append (str(Path(__file__ ).parent))

class FRDataConfig:
    N_CLASSES = 5749
    IMG_SIZE = 64
    ID2DLABEL = {}
    LABEL2ID = {}
    NORMALIZE_MEAN = [0.5 , 0.5 , 0.5]
    NORMALIZE_STD = [0.5 , 0.5, 0.5]
    
    @classmethod
    def update_label_mappings(cls, excel_path):
        # Đọc file Excel
        df = pd.read_excel(excel_path)

        # Giả định bảng có 2 cột: "Label" và "ID"
        cls.ID2DLABEL = dict(zip(df['ID'], df['Label']))
        cls.LABEL2ID = dict(zip(df['Label'], df['ID']))
        
excel_file = Path('Name.xlsx')
FRDataConfig.update_label_mappings(excel_file)

class ModelConfig:
    ROOT_DIR = Path ( __file__ ).parent.parent
    MODEL_NAME = 'resnet34'
    MODEL_WEIGHT = ROOT_DIR/'models'/'weights'/'FR_weights.pt'
    DEVICE = 'cpu'