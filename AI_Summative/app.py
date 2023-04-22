import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("lgb_tuned_model.pkl", "rb"))

# Define homepage route
@flask_app.route("/")
def home():
    return render_template("home.html")

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("lgb_tuned_model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():

    features_all = [x for x in request.form.values()]

    # Map 'True' to 1 and 'False' to 0
    mapping = {'True': 1, 'False': 0}
    all_features = [mapping[feature]
                    if feature in mapping else feature for feature in features_all]
    # numerical_features = [
    #     feature for feature in all_features if (type(feature) == int or type(feature) == float)]

    # extract the numerical features from user input 
    numerical_features = all_features[6:13]

    # store the numerical features converted to int or float 
    conv_numerical_features = []

    # string to int/float conversion
    for i in numerical_features:
        is_Int = False
        try:
            int(i)
            is_Int = True
        except ValueError:
            is_Int = False

        if is_Int:
            conv_numerical_features.append(int(i))
        else:
            conv_numerical_features.append(float(i))

    # conv_numerical_features = np.array(conv_numerical_features)

    # num_types = [type(i) for i in conv_numerical_features]

    # Extract categorical features
    # categorical_features = [
    #     feature for feature in all_features if not (type(feature) == int or type(feature) == float)]

    # extract the categorical features from user input
    categorical_features = all_features[0:6]

    # extract the boolean features from user input
    boolean_features = all_features[13:]

    # Initialize the LabelEncoder
    encoder = LabelEncoder()

    # Fit and transform the categorical features
    encoded_features = encoder.fit_transform(categorical_features)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the numerical features
    scaled_features = scaler.fit_transform([conv_numerical_features])

    # Concatenate encoded categorical features, encoded numerical features and boolean features
    features = np.concatenate(
        (encoded_features, boolean_features, scaled_features[0]), axis=0)
    features = [np.array(features)]

    # Make the prediction
    prediction = model.predict(features)

    # Return the result to the user
    return render_template("index.html", prediction_text=" Malware result: {}".format(prediction))

    # # Return the result to the user
    # return render_template("index.html", prediction_text=" Malware result: {}".format(prediction.round(0).astype(int)) if prediction==0 then prediction= "You don't have malwre" else prediction ="You have malware")

    # Return the result to the user
    # return render_template("index.html", prediction_text="Malware result: {}\n\nValues entered: {}\n\nCategorical features: {}\n\nBoolean features: {}\n\nNumerical features: {}".format(prediction, features_all, categorical_features, boolean_features, conv_numerical_features))

if __name__ == "__main__":
    flask_app.run(debug=True)