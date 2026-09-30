from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------------- TRAINING DATA ----------------

issues = [

    # NETWORK
    "wifi is not working",
    "wifi keeps disconnecting",
    "internet is not working",
    "internet connection is slow",
    "laptop cannot connect to wifi",
    "network connection keeps dropping",
    "ethernet is not working",
    "network is unavailable",
    "vpn is not connecting",
    "cannot connect to office network",
    "network connection error",
    "internet keeps disconnecting",
    "office wifi is not connecting",
    "company network is unavailable",
    "my laptop has no internet",
    "unable to connect to corporate wifi",
    "internet connection stopped working",
    "vpn connection keeps failing",
    "network keeps disconnecting",
    "office internet is unavailable",
    "cannot access the internet",
    "wifi connection failed",
    "network access is not working",
    "internet is disconnected",
    "unable to connect to wifi",

    # ACCESS / LOGIN
    "forgot my password",
    "cannot login to my account",
    "my account is locked",
    "login is not working",
    "authentication failed",
    "cannot access my account",
    "password reset is not working",
    "username and password are not accepted",
    "unable to sign in",
    "employee account access problem",
    "login credentials are not working",
    "access denied to my account",
    "forgot login password",
    "account is locked",
    "cannot sign into my account",
    "unable to access employee account",
    "password is not accepted",
    "login credentials failed",
    "cannot access company account",
    "my account access is blocked",
    "sign in is not working",
    "unable to login",
    "password reset failed",
    "employee login problem",

    # HARDWARE
    "laptop is not turning on",
    "keyboard is not working",
    "mouse is not working",
    "printer is not working",
    "monitor has no display",
    "computer hardware problem",
    "laptop screen is broken",
    "laptop battery is not charging",
    "printer is showing an error",
    "keyboard keys are not responding",
    "computer is overheating",
    "monitor is not connecting",
    "laptop will not start",
    "computer is not powering on",
    "mouse is not responding",
    "printer cannot print",
    "monitor is blank",
    "laptop screen is damaged",
    "battery is not charging",
    "keyboard is broken",
    "printer hardware problem",
    "computer is overheating",
    "external monitor is not working",
    "laptop hardware issue",

    # SOFTWARE
    "application is crashing",
    "software is not working",
    "program keeps showing an error",
    "application will not open",
    "software update failed",
    "program is running slowly",
    "application stopped responding",
    "software installation failed",
    "application shows an error",
    "program is not responding",
    "software keeps crashing",
    "cannot install the application",
    "application keeps crashing",
    "program will not open",
    "software is running slowly",
    "application is frozen",
    "application stopped working",
    "software installation error",
    "application update failed",
    "program is showing an error",
    "cannot open the software",
    "software is not responding",
    "application installation failed",
    "program keeps crashing",

    # OTHER

    "I need technical support",
    "I need IT support",
    "I need general IT assistance",
    "I have a general IT question",
    "I need help from IT support",
    "please contact IT support",
    "I want to raise an IT support request",
    "I need assistance from the IT team",
    "I have a general technical question",
    "please help me with an IT request",
    "I need help with an IT service",
    "I want to contact the IT helpdesk"
]

categories = (
    ["Network"] * 25
    + ["Access / Login"] * 24
    + ["Hardware"] * 24
    + ["Software"] * 24
    + ["Other"] * 12
)

# ---------------- TEXT VECTORIZATION ----------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(issues)


# ---------------- TRAIN MODEL ----------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, categories)


# ---------------- PREDICTION ----------------

def predict_category(issue):

    issue_vector = vectorizer.transform([issue])

    prediction = model.predict(issue_vector)[0]

    probability = max(
        model.predict_proba(issue_vector)[0]
    )

    confidence = round(probability * 100, 2)

    return prediction, confidence


# ---------------- TEST MODEL ----------------

if __name__ == "__main__":

    test_issues = [
        "My laptop cannot connect to WiFi",
        "I forgot my password",
        "My printer is not working",
        "The application keeps crashing",
        "I need technical support"
    ]

    for issue in test_issues:

        category, confidence = predict_category(issue)

        print(f"Issue: {issue}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence}%")
        print("-" * 50)