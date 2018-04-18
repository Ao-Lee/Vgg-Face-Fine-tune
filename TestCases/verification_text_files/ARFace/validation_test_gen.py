import numpy as np
import pandas as pd

filename = 'validation.txt'
suffix = 'bmp'

def CreateName(sex, personID, imageID):
    name = sex + '-' + '%03d'%personID + '-' + '%02d'%imageID
    return name + '.' + suffix

def GenPos():
    list_first = []
    list_second = []
    for idx_sex in ['M','W']:
        for idx_person in range(1, 51):
            for idx_img in range(2, 27):
                name_first = CreateName(idx_sex, idx_person, 1)
                name_second = CreateName(idx_sex, idx_person, idx_img)
                list_first.append(name_first)
                list_second.append(name_second)
    df = pd.DataFrame([])
    df['img1'] = pd.Series(list_first)
    df['img2'] = pd.Series(list_second)
    df['class'] = 'S'
    return df
    
def GenNeg():
    list_first = []
    list_second = []
    np.random.seed(231)
    for _ in range(2500):
        sex = 'M' if np.random.randint(low=0, high=2, size=1)[0]==0 else 'W'
        person_a = np.random.randint(low=1, high=51, size=1)[0]
        while True:
            person_b = np.random.randint(low=1, high=51, size=1)[0]
            if person_a != person_b:
                break
            
        img = np.random.randint(low=2, high=27, size=1)[0]
        name_first = CreateName(sex, person_a, 1)
        name_second = CreateName(sex, person_b, img)
        list_first.append(name_first)
        list_second.append(name_second)
        assert person_a>=1 and person_a<=50
        assert person_b>=1 and person_b<=50
        assert person_a!=person_b
        assert img>=2 and img<=26

    df = pd.DataFrame([])
    df['img1'] = pd.Series(list_first)
    df['img2'] = pd.Series(list_second)
    df['class'] = 'D'
    return df

if __name__=='__main__':
    df1 = GenPos()
    df2 = GenNeg()
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(filename, sep='\t', index=False, header=True)

    
                
