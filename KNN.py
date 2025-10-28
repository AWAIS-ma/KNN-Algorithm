import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn(animal, sym1, sym2, sym3):

    data = pd.read_csv("animal_disease_dataset_up.csv")

    label_encoders = {}
    for col in ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Disease']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le


    X = data[['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3']]
    y = data['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 5
    max_k = 20
    model = None
    confidence = 0

    try:
        animal_code = label_encoders['Animal'].transform([animal])[0]
        s1_code = label_encoders['Symptom 1'].transform([sym1])[0]
        s2_code = label_encoders['Symptom 2'].transform([sym2])[0]
        s3_code = label_encoders['Symptom 3'].transform([sym3])[0]
    except ValueError:
        return {"error": "Invalid input. One or more entries not found in training data."}

    input_data = pd.DataFrame([[animal_code, s1_code, s2_code, s3_code]],
                              columns=['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3'])

    while k <= max_k:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100

        
        prediction = knn_model.predict(input_data)
        probabilities = knn_model.predict_proba(input_data)
        confidence = probabilities.max() * 100

        if confidence >= 80:
            model = knn_model
            break
        k += 1

    predicted_disease = label_encoders['Disease'].inverse_transform(prediction)[0]

    return {
        "Predicted Disease": predicted_disease,
        "Confidence": round(confidence, 2),
        "Model Accuracy": round(acc, 2),
    }
