import pandas, shutil, os

# df_I = pandas.read_csv(r"D:\k-ium\train_I.csv")
df_V = pandas.read_csv(r"D:\k-ium\train_V.csv")

os.makedirs('D:/k-ium/train_set/V/A/Normal', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/A/VA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/A/PICA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/A/SCA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/A/BA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/A/PCA', exist_ok=True)

os.makedirs('D:/k-ium/train_set/V/B/Normal', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/B/VA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/B/PICA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/B/SCA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/B/BA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/V/B/PCA', exist_ok=True)

for row in df_V.iterrows():
    patient_id = row[1]['Index']
    Aneurysm = row[1]['Aneurysm']
    VA = row[1]['VA']
    PICA = row[1]['PICA']
    SCA = row[1]['SCA']
    BA = row[1]['BA']
    PCA = row[1]['PCA']
    
    if Aneurysm ==0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/Normal', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/Normal', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/Normal', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/Normal', f'{patient_id}RI-B.jpg'))
        
    if VA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/VA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/VA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/VA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/VA', f'{patient_id}RI-B.jpg'))
    
    if PICA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/PICA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/PICA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/PICA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/PICA', f'{patient_id}RI-B.jpg'))
        
    if SCA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/SCA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/SCA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/SCA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/SCA', f'{patient_id}RI-B.jpg'))
    
    if BA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/BA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/BA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/BA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/BA', f'{patient_id}RI-B.jpg'))
    
    if PCA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/PCA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/PCA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/V/A/PCA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/V/B/PCA', f'{patient_id}RI-B.jpg'))