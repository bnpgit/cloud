from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import sqlite3
import os
import random
import secrets
from model import predict_priority
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(16)

# Initialize database
def init_db():
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS user(
        name TEXT, 
        password TEXT, 
        mobile TEXT, 
        email TEXT
    )""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS sensors(
        time TEXT, 
        voltage TEXT, 
        current TEXT, 
        temp TEXT, 
        humidity TEXT, 
        Ac_voltage TEXT
    )""")
    
    connection.commit()
    connection.close()

init_db()

# Load dataset for visualization
df = pd.read_csv('task_dataset.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))
        result = cursor.fetchone()

        if result:
            session['name'] = name
            return render_template('userlog.html')
        else:
            return render_template('signin.html', msg='Sorry, Incorrect Credentials Provided, Try Again')

    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", 
                      (name, password, mobile, email))
        connection.commit()

        return render_template('signin.html', msg='Successfully Registered')
    
    return render_template('signup.html')

@app.route('/analysis')
def analysis_page():
    # This will render the analysis HTML page
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'tasks' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
            
        predictions = []
        
        for task in data['tasks']:
            try:
                execution_time = float(task['execution_time'])
                cpu_req = float(task['cpu_req'])
                memory_req = float(task['memory_req'])
                storage_req = float(task['storage_req'])
                data_transfer_size = float(task['data_transfer_size'])
                
                priority, confidence = predict_priority(
                    execution_time, cpu_req, memory_req, 
                    storage_req, data_transfer_size
                )
                
                predictions.append({
                    'priority': int(priority),
                    'confidence': float(confidence)
                })
            except (KeyError, ValueError) as e:
                return jsonify({'error': f'Invalid task data: {str(e)}'}), 400
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            df = pd.read_csv('data.csv')
            selected_rows = df.sample(3)
            
            # Your prediction logic here
            # For example:
            # mod = pickle.load(open("svm_model.pkl","rb"))
            # selected_rows = mod.predict([[60.42133853068739,544.0240108259324,15.270393997485542]])
            
            min_index = selected_rows.iloc[:, 2].idxmin()
            
            tasks = []
            for idx, row in selected_rows.iterrows():
                task = {
                    'col1': {'name': df.columns[0], 'value': row[0]},
                    'col2': {'name': df.columns[1], 'value': row[1]},
                    'col3': {'name': df.columns[2], 'value': row[2]},
                    'col4': {'name': df.columns[3], 'value': row[3]},
                    'col5': {'name': df.columns[4], 'value': row[4]},
                    'is_immediate': idx == min_index
                }
                tasks.append(task)
            
            return render_template('userlog.html', tasks=tasks)
        except Exception as e:
            return render_template('userlog.html', error=str(e))
    return render_template('userlog.html')

@app.route('/get_task_data')
def get_task_data():
    sample_df = df.sample(n=20)
    tasks = []
    for _, row in sample_df.iterrows():
        tasks.append({
            'task_id': row['task_id'],
            'task_name': row['task_name'],
            'execution_time': row['execution_time'],
            'cpu_req': row['cpu_req'],
            'memory_req': row['memory_req'],
            'storage_req': row['storage_req'],
            'data_transfer_size': row['data_transfer_size'],
            'priority': row['priority'],
            'task_type': row['task_type'],
            'sla_critical': row['sla_critical'],
            'status': random.choice(['Pending', 'Running', 'Completed', 'Failed'])
        })
    return jsonify(tasks)

@app.route('/get_stats')
def get_stats():
    stats = {
        'total_tasks': len(df),
        'avg_execution_time': df['execution_time'].mean(),
        'priority_distribution': df['priority'].value_counts().to_dict(),
        'task_type_distribution': df['task_type'].value_counts().to_dict(),
        'sla_critical_percentage': (df['sla_critical'].sum() / len(df)) * 100
    }
    return jsonify(stats)

@app.route('/logout')
def logout():
    session.pop('name', None)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
