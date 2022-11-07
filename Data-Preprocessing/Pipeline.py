from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, random_state=0)
# fit the whole pipeline
pipe.fit(X_train, y_train)

print('printing training accuracy')
print(accuracy_score(pipe.predict(X_validate), y_validate))


# we can now use it like any other estimator
print('printing testing accuracy')
print(accuracy_score(pipe.predict(X_test), y_test))