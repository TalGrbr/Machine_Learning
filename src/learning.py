import numpy as np
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import get_data


def svm_params():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    c_options = [0.001, 0.01, 0.1, 1, 10]
    gammas_options = [0.001, 0.01, 0.1, 1, 'auto']
    kernels = ['linear', 'rbf', 'sigmoid']
    for kernel in kernels:
        param_grid = {'C': c_options, 'gamma': gammas_options}
        grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        print(kernel)
        print(grid_search.best_params_)


def svm():
    x, y = get_data()
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    svc_classifier = SVC(C=100, gamma=1, kernel='rbf', shrinking=False, decision_function_shape='ovo', random_state=42)
    svc_classifier.fit(x_train, y_train)
    y_pred = svc_classifier.predict(x_test)
    print('SVM Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('SVM classification report')
    print(classification_report(y_test, y_pred))
    return y_pred


def rfc_params():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 150, num=11)]
    max_depth.append(None)
    min_samples_split = [1, 2, 5, 10, 12, 15]
    min_samples_leaf = [1, 2, 4, 6]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    rf = RandomForestClassifier()
    # Random search of parameters
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=35, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)


def rfc():
    x, y = get_data()
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    rfc_model = RandomForestClassifier(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                       max_depth=150, bootstrap=True)
    rfc_model.fit(x_train, y_train)
    y_pred = rfc_model.predict(x_test)
    print('RFC confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('RFC classification report')
    print(classification_report(y_test, y_pred))
    return y_pred


def compare_svm_rfc():
    x, y = get_data()
    svm_results = svm()
    rfc_results = rfc()

    diff = [s - r for s, r in zip(svm_results, rfc_results)]

    d_bar = np.mean(diff)
    sigma2 = np.var(diff)
    n1 = 0.8 * x.shape[0]
    n2 = 0.2 * x.shape[0]
    n = x.shape[0]
    sigma2_mod = sigma2 * (1 / n + n2 / n1)
    t_static = d_bar / np.sqrt(sigma2_mod)
    p_value = np.abs(1 - (t.cdf(t_static, n - 1)) * 2)
    print('SVM and RFC t-test p_value result: ' + str(p_value))
