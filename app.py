import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    ################ Step 1 Create Web Title #####################

    st.title("Binary Classification Streamlit App")
    st.sidebar.title("Binary Classification Streamlit App")
    st.markdown(" เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")

    st.markdown("### คำอธิบายเกี่ยวกับโมเดล")
    st.markdown("""
    - **Support Vector Machine (SVM)**: โมเดลสำหรับการแบ่งประเภทที่ใช้เส้นขอบเขต (hyperplane) เพื่อแบ่งกลุ่มข้อมูล
    - **Logistic Regression**: โมเดลที่ใช้สำหรับการพยากรณ์ความน่าจะเป็น โดยให้ผลลัพธ์เป็นค่าระหว่าง 0 ถึง 1
    - **Random Forest**: โมเดลที่รวมการทำงานของหลายๆ Decision Trees เพื่อเพิ่มความแม่นยำในการแบ่งประเภท
    """)

    st.markdown("### คำแนะนำการเลือกพารามิเตอร์")
    st.markdown("""
    - **C (SVM & Logistic Regression)**: ควบคุมการ regularization; ค่าสูงช่วยให้โมเดลฟิตข้อมูลได้มาก ค่าต่ำช่วยลดความซับซ้อนของโมเดล
    - **Kernel (SVM)**: เลือกรูปแบบ kernel; ใช้ "linear" สำหรับข้อมูลที่มีการแยกเชิงเส้น หรือ "rbf" สำหรับข้อมูลที่ซับซ้อน
    - **Gamma (SVM)**: ควบคุมการโค้งของขอบเขตใน kegit init
    git remote add origin https://github.com/yourusername/yourrepositoryname.git
     ค่าสูงสำหรับรายละเอียดมาก ค่าต่ำสำหรับการป้องกัน overfitting
    - **n_estimators (Random Forest)**: จำนวนต้นไม้; จำนวนมากขึ้นทำให้โมเดลแม่นยำขึ้น แต่ใช้ทรัพยากรเพิ่มขึ้น
    - **max_depth (Random Forest)**: ความลึกสูงสุดของต้นไม้; ค่าสูงทำให้โมเดลฟิตข้อมูลมาก ค่าต่ำช่วยป้องกัน overfitting
    """)

    st.sidebar.markdown(" เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")

    ############### Step 2 Load dataset and Preprocessing data ##########

    # create read_csv function
    @st.cache_data(persist=True) #ไม่ให้ streamlit โหลดข้อมูลใหม่ทุกๆครั้งเมื่อ run
    def load_data():
        # Define the base directory as the folder containing your script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file_path = os.path.join(DATA_DIR, 'mushrooms.csv')

        data = pd.read_csv(file_path)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def spliting_data(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=class_names)
            st.pyplot(fig)
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
      
    df = load_data()
    x_train, x_test, y_train, y_test = spliting_data(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifiers")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    ############### Step 3 Train a SVM Classifier ##########

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)

    ############### Step 4 Train a Logistic Regression Classifier ##########
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify_LR'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)

    ############### Step 5 Train a Random Forest Classifier ##########
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Max depth of the tree", 1, 20, step=1, key='max_depth')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify_RF'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset")
        st.write(df)

if __name__ == '__main__':
    main()
