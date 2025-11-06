import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn(animal, sym1, sym2, sym3):
    data = pd.read_csv("expanded_animal_disease_dataset.csv").dropna().drop_duplicates()

    X = data[['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3']]
    y = data['Disease']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = encoder.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_df = pd.DataFrame([[animal, sym1, sym2, sym3]], columns=X.columns)
    input_encoded = encoder.transform(input_df)
    input_scaled = scaler.transform(input_encoded)

    params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['distance', 'uniform'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
    grid.fit(X_train, y_train)
    knn_model = grid.best_estimator_

    y_pred = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    prediction = knn_model.predict(input_scaled)
    probs = knn_model.predict_proba(input_scaled)
    confidence = probs.max() * 100

    predicted_disease = label_encoder.inverse_transform(prediction)[0]

    return {
        "Predicted Disease": predicted_disease,
        "Confidence": round(confidence, 2),
        "Model Accuracy": round(acc, 2),
        "Best Params": grid.best_params_
    }
