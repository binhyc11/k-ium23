import pandas, shutil, os

df_I = pandas.read_csv(r"D:\k-ium\train_I.csv")

os.makedirs('D:/k-ium/train_set/I/A/Normal', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/A/ICA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/A/PCOM', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/A/AntChor', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/A/ACA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/A/ACOM', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/A/MCA', exist_ok=True)

os.makedirs('D:/k-ium/train_set/I/B/Normal', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/B/ICA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/B/PCOM', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/B/AntChor', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/B/ACA', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/B/ACOM', exist_ok=True)
os.makedirs('D:/k-ium/train_set/I/B/MCA', exist_ok=True)

for row in df_I.iterrows():
    patient_id = row[1]['Index']
    Aneurysm = row[1]['Aneurysm']
    ICA = row[1]['ICA']
    PCOM = row[1]['PCOM']
    AntChor = row[1]['AntChor']
    ACA = row[1]['ACA']
    ACOM = row[1]['ACOM']
    MCA = row[1]['MCA']
    
    if Aneurysm ==0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/Normal', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/Normal', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/Normal', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/Normal', f'{patient_id}RI-B.jpg'))
        
    if ICA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/ICA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/ICA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/ICA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/ICA', f'{patient_id}RI-B.jpg'))
    
    if PCOM >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/PCOM', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/PCOM', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/PCOM', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/PCOM', f'{patient_id}RI-B.jpg'))
        
    if AntChor >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/AntChor', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/AntChor', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/AntChor', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/AntChor', f'{patient_id}RI-B.jpg'))
    
    if ACA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/ACA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/ACA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/ACA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/ACA', f'{patient_id}RI-B.jpg'))
    
    if ACOM >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/ACOM', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/ACOM', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/ACOM', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/ACOM', f'{patient_id}RI-B.jpg'))
        
    if MCA >0:
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/MCA', f'{patient_id}LI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}LI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/MCA', f'{patient_id}LI-B.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-A.jpg'), 
            os.path.join('D:/k-ium/train_set/I/A/MCA', f'{patient_id}RI-A.jpg'))
        shutil.copy(os.path.join('D:/k-ium/preprocessing', f'{patient_id}RI-B.jpg'), 
            os.path.join('D:/k-ium/train_set/I/B/MCA', f'{patient_id}RI-B.jpg'))