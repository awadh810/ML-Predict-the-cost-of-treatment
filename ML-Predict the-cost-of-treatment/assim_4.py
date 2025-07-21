#----------------------------------------- <<  Import libs that we need to build this project >> --------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

#!-------------------------------------------- <<  Reading Data From csv File >> ------------------------------------------------
date_formats = ['%d-%m-%Y', '%Y/%m/%d', '%m/%d/%Y', '%Y-%m-%d']
data = pd.read_csv("./hospital_records_2021_2024_with_bills.csv")

#?-------------------------------------------------- <<  Data Cleaning  >> -----------------------------------------------
def data_cleaning(data_Para):    
    #! Handle missing values    
    print(data_Para.isnull().sum())   
    data_Para.fillna(
    {   #! إعداد القيم الافتراضية لكل عمود في حالة وجود خلايا فارغه
        'Medical Condition': 'undefined',
        'Treatments': 'undefined',
        "Doctor's Notes": 'undefined',
        'Gender': data_Para['Gender'].mode()[0], 
        'Bill Amount': data_Para['Bill Amount'].mode()[0]
    }, inplace=True)    

    #? Process the Dates in Wrong Format
    date_cols = ['Date of Birth', 'Admit Date', 'Discharge Date']    
    for col in date_cols:        
        data_Para[col] = data_Para[col].apply(convert_date)      
        #data_Para[col] = data_Para[col].dt.strftime('%Y/%m/%d')

    #!Process The Duplicates Rows by delete it 
    #print(data_Para.duplicated(keep=False).to_string())
    data_Para = data_Para.drop_duplicates(keep='first') 
    data_Para = data_Para.reset_index(drop=True)     

    return data_Para    # ? return final data after process it

#?------------------------------------------------ << Convert Date To object Type>> ------------------------------------------------
def convert_date(date_str):
    for format in date_formats:  
        try:
            return pd.to_datetime(date_str, format=format)
        except ValueError:
            continue
    return pd.NaT  
 
#!---------------------------------------------- << Outliers Detection & Removal >> -------------------------------------------
def Outliers_Detection_Removal(data_para):
    plt.scatter(data_para.index , data_para['Bill Amount'] , color='g', label='amount')
    plt.legend()
    plt.show()

    # تحليل القيم الشاذة باستخدام IQR لعمود "Bill Amount"
    Q1 = data_para['Bill Amount'].quantile(0.25)
    Q3 = data_para['Bill Amount'].quantile(0.75)
    IQR = Q3 - Q1
    
    #plt.scatter(data_para.index , data_para['Bill Amount'] , color='g', label='amount')
    # حدود القيم الشاذة
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_outliers = data_para[((data_para['Bill Amount'] < lower_bound) | (data_para['Bill Amount'] > upper_bound))]    
    data_cleaned = data_para[~((data_para['Bill Amount'] < lower_bound) | (data_para['Bill Amount'] > upper_bound))]   # إزالة القيم الشاذة

    #print(data_outliers)
    return data_cleaned , data_outliers

#?------------------------------------------------  << Draw Data  >>  -------------------------------------------------------
def draw_data(df):
    # إعداد الشكل
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # رسم عدد المرضى حسب الجنس
    axs[0].bar(df['Gender'].value_counts().index, df['Gender'].value_counts().values, color=['blue', 'orange'])
    axs[0].set(title='Number of Patients by Gender', ylabel='Number of Patients', xlabel='Gender')

    # رسم توزيع الحالات الطبية
    explode = [0.2] * len(df['Medical Condition'].value_counts())  # يبعد كل قطعة قليلاً
    axs[1].pie(df['Medical Condition'].value_counts(), labels=df['Medical Condition'].value_counts().index, autopct= "%.1f%%", startangle=90, explode=explode)
    axs[1].axis('equal')  # يجعل الرسم دائريا
    axs[1].set_title('Distribution of Medical Conditions')
    
    
    # عرض الرسم
    plt.tight_layout()
    plt.show()

#!-----------------------------------------------  << Training ML Model >> -----------------------------------------------------
                                                    # todoo >>>  توقع تكاليف العلاج
def train_model_ml(df):  
    
    df['Length of Stay'] = (df['Discharge Date'] - df['Admit Date']).dt.days        # حساب مدة الإقامة في المستشفى (Length of Stay)
    df['Age'] = (pd.to_datetime('today') - df['Date of Birth']).dt.days // 365      # استخراج العمر من تاريخ الميلاد
     
    features = df[['Age', 'Length of Stay', 'Gender', 'Medical Condition', 'Treatments']]   # اختيار الميزات (Features)
    target = df['Bill Amount']      # اختيار الهدف (Target)

    # تحويل الميزات الفئوية او الاسمية إلى أرقام باستخدام Label Encoding
    label_encoders = {}  # في حال اردت ارجع الارقام الى الفئات الاصلية كاسماء  
    for column in ['Gender', 'Medical Condition', 'Treatments']:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le        

    scaler = StandardScaler()      # تقييس الميزات العددية
    features[['Age', 'Length of Stay']] = scaler.fit_transform(features[['Age', 'Length of Stay']])

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)    # تقسيم البيانات إلى مجموعة تدريب واختبار
    
    model = RandomForestRegressor(random_state=42)          # n_estimators=100,   # بناء النموذج باستخدام Random Forest Regressor    استخدم خوارزمية اخرى
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)     #  توقع تكاليف العلاج على مجموعة الاختبار
    
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})   # عرض التوقعات
    print("The Prediction for Bill Amount Prediction are:  \n" , results)

    # حساب RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))          
    mean_actual = np.mean(y_test)   # حساب متوسط القيم الحقيقية    
    rmse_percentage = (rmse / mean_actual) * 100    # حساب النسبة المئوية لـ RMSE

    # طباعة RMSE
    print(f'Root Mean Squared Error (RMSE): {int(rmse)}')  # طباعة RMSE كنسبة مئوية
    print(f'RMSE Percentage: {int(rmse_percentage)}%')  # طباعة النسبة المئوية لـ RMSE


# todoo =============================================== << Main FUN. To excecute Tasks >> =======================================================
def main():  
    data_cleaned = data_cleaning(data)
    #print('------------------------------------- << List Of Data After Clean it >> -----------------------------------------------------')
    #print(data_cleaned.to_string() , '\n\n') 

# ! --------------------------------------------------   ***  --------------------------------------

    data_with_outliers , data_outliers = Outliers_Detection_Removal(data_cleaned)
    #print('------------------------------------- << List Of Data Without Outliers >> --------------------------------------------------')
    #print(data_with_outliers , '\n')
    #print('------------------------------------- << List Of Data With Outliers >> -----------------------------------------------------')
    #print(data_outliers , '\n\n')

# ! --------------------------------------------------   ***  -------------------------------------

    #print('----------------------------------- << Drawing Of Data Without Outliers and After Clean it >> ------------------------------')
    draw_data(data_with_outliers)

# ? -------------------------------------------------   ***  ---------------------------------------

    #print('----------------------------------- << Traing ML Model >> ------------------------------')
    train_model_ml(data_with_outliers)

# ! <><><><<><><><><><><><><><><><><><><>< Calling Main FUN. ><><><><><><><><><><><><><><><><><><><><><>
if __name__ == "__main__":
    main()