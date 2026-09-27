\# AI Enterprise HelpDesk



Smart IT Support \& Ticket Management



AI Enterprise HelpDesk is a Flask-based enterprise IT support application that helps employees submit IT issues, automatically classify them using machine learning, determine priority, receive troubleshooting recommendations, and create support tickets.



\## Key Features



\- IT issue classification using NLP and machine learning

\- TF-IDF text feature extraction

\- Logistic Regression classification

\- Automatic issue categories:

&#x20; - Network

&#x20; - Access / Login

&#x20; - Hardware

&#x20; - Software

&#x20; - Other

\- Automatic priority detection

\- Category-based troubleshooting recommendations

\- Support ticket creation

\- Ticket status tracking

\- Admin support dashboard

\- Ticket resolution management

\- Local SQLite database

\- Runs locally without external APIs



\## Technology Stack



\- Python

\- Flask

\- Scikit-learn

\- SQLite

\- HTML

\- CSS

\- JavaScript



\## How It Works



Employee submits an IT issue



↓



NLP-based ML classifier analyzes the issue



↓



TF-IDF converts the text into numerical features



↓



Logistic Regression predicts the issue category



↓



System determines ticket priority



↓



Troubleshooting recommendation is displayed



↓



Employee creates a support ticket



↓



Admin manages and resolves the ticket



\## Project Structure



```text

AI-Enterprise-HelpDesk

├── app.py

├── ml\_classifier.py

├── templates

│   ├── index.html

│   ├── dashboard.html

│   └── ticket\_success.html

├── static

│   └── style.css

└── .gitignore

