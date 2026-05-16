from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def get_models():
    return {
        "logreg": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
        ),

        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),
            random_state=RANDOM_STATE,
        ),

        "gaussian_process": GaussianProcessClassifier(
            random_state=RANDOM_STATE,
        ),

        "gaussian_nb": GaussianNB(),

        "knn": KNeighborsClassifier(),

        "svc": SVC(random_state=RANDOM_STATE),

        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
    }