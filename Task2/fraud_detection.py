import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\Nikita Kodgire\Desktop\task2codsoft\fraudTrain.csv")  # Make sure this file is in the same folder

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street',
             'city', 'state', 'job', 'dob', 'merchant', 'trans_num']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split dataset
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:")
print(report)

# Save model
joblib.dump(model, "model.pkl")
print("\n💾 Model saved as model.pkl")
