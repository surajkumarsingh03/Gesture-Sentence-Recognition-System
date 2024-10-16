pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the repository
                git url: 'https://github.com/surajkumarsingh03/Gesture-Sentence-Recognition-System'
            }
        }

        stage('Build') {
            steps {
                // Install dependencies (adjust based on your project's needs)
                sh ' python app.py'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Deploy') {
            steps {
                // Steps for deployment (optional)
                echo 'Deploying application...'
            }
        }
    }
}
