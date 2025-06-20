from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

csv_path = 'Original_Parkinsondataset1.csv'
model = None
current_df = None  # will switch to uploaded file if available

# Model Training
try:
    df = pd.read_csv(csv_path)
    X = df.drop(['status', 'Sr.No.'], axis=1)
    y = df['status']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    current_df = df.copy()
except Exception as e:
    print("Error in model training:", e)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    global current_df
    data = request.get_json()
    sr_no_input = data.get('person', '').strip()

    if not sr_no_input.isdigit():
        return jsonify({'error': 'Enter valid Sr.No. (numeric).'})

    sr_no = int(sr_no_input)
    try:
        df = current_df
        row = df[df['Sr.No.'] == sr_no]
        if row.empty:
            return jsonify({'error': 'Sr.No. not found.'})
        status = row['status'].values[0]
        result = "Disease Detected" if status == 1 else "No Disease Detected"

        # Generate graphs for the current file
        X = df.drop(['status', 'Sr.No.'], axis=1)
        y = df['status']
        predictions = model.predict(X)

        cm = confusion_matrix(y, predictions)
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')

        plt.subplot(1,3,2)
        fpr, tpr, _ = roc_curve(y, predictions)
        plt.plot(fpr, tpr, marker='.')
        plt.title('ROC Curve')

        plt.subplot(1,3,3)
        features = X.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
        plt.title('Top 10 Feature Importances')

        graph_path = os.path.join('static', 'output.png')
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        return jsonify({'result': result, 'graph_path': graph_path})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            new_df = pd.read_csv(filepath)
            new_X = new_df.drop(['status', 'Sr.No.'], axis=1)
            new_y = new_df['status']
            model.fit(new_X, new_y)
            current_df = new_df.copy()

            # Generate plots for new data
            predictions = model.predict(new_X)
            cm = confusion_matrix(new_y, predictions)
            plt.figure(figsize=(15,5))

            plt.subplot(1,3,1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')

            plt.subplot(1,3,2)
            fpr, tpr, _ = roc_curve(new_y, predictions)
            plt.plot(fpr, tpr, marker='.')
            plt.title('ROC Curve')

            plt.subplot(1,3,3)
            features = new_X.columns
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
            plt.title('Top 10 Feature Importances')

            graph_path = os.path.join('static', 'output.png')
            plt.tight_layout()
            plt.savefig(graph_path)
            plt.close()

            return jsonify({'message': 'File processed.', 'graph_path': graph_path})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
