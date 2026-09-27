from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Training examples
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

    # OTHER
    "I need technical support",
    "I need help with an IT issue",
    "there is a technical problem",
    "I have another IT problem",
    "I need IT assistance",
    "technical support is required",
    "I need help with my computer",
    "please help with an IT problem"
]


categories = [

    # NETWORK
    "Network", "Network", "Network", "Network",
    "Network", "Network", "Network", "Network",
    "Network", "Network", "Network", "Network",

    # ACCESS / LOGIN
    "Access / Login", "Access / Login", "Access / Login",
    "Access / Login", "Access / Login", "Access / Login",
    "Access / Login", "Access / Login", "Access / Login",
    "Access / Login", "Access / Login", "Access / Login",

    # HARDWARE
    "Hardware", "Hardware", "Hardware", "Hardware",
    "Hardware", "Hardware", "Hardware", "Hardware",
    "Hardware", "Hardware", "Hardware", "Hardware",

    # SOFTWARE
    "Software", "Software", "Software", "Software",
    "Software", "Software", "Software", "Software",
    "Software", "Software", "Software", "Software",

    # OTHER
    "Other", "Other", "Other", "Other",
    "Other", "Other", "Other", "Other"
]


# TF-IDF converts text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(issues)


# Train the classifier
model = LogisticRegression(
    max_iter=1000
)

model.fit(X, categories)


def predict_category(issue):

    issue_vector = vectorizer.transform([issue])

    prediction = model.predict(issue_vector)[0]

    probability = max(
        model.predict_proba(issue_vector)[0]
    )

    confidence = round(probability * 100, 2)

    return prediction, confidence


# Test the model
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

        print(
            f"Issue: {issue}"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Confidence: {confidence}%"
        )

        print("-" * 50)