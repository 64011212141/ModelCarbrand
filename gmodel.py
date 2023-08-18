import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

#การแยก x กับ y ดหลดข้อมูลจากไฟล์จาก train=ให้เรียนรู้ฝึก test=ทดสอบ 
carvectors_train = pickle.load(open('carvectors_train.pkl', 'rb'))
carvectors_test = pickle.load(open('carvectors_test.pkl', 'rb'))

#แบ่งสัดส่วน x y y=class ในช่วงของ 0-8100 มาเก็บไว้ใน x_test
X_train_data = [carvectors_train[0:8100] for  carvectors_train in carvectors_train]
X_test_data = [carvectors_test[0:8100] for carvectors_test in carvectors_test]
#-1  เอามาจากตัวสุดท้าย
Y_train_data = [carvectors_train[-1] for carvectors_train in carvectors_train]
Y_test_data = [carvectors_test[-1] for carvectors_test in carvectors_test]
#เป้นการแปลงตัวหนังสือเป้นตัวเลข
LE = LabelEncoder()
new_y_train= LE.fit_transform(Y_train_data)
new_y_test= LE.fit_transform(Y_test_data)
#สร้าง model Decision tree
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train_data, new_y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_data)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(new_y_test, y_pred)
accuracy_percentage = accuracy * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")


print(confusion_matrix(new_y_test,y_pred))
write_path = "model.pkl"
pickle.dump(clf, open(write_path,"wb"))
print("data preparation is done")
