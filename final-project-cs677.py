"""
Author: Wasupol Tungsakultong
Course: CS677 - Term Project
DataSet: English premier league
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import tree, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

df_2014_2015, df_2015_2016, df_2016_2017, df_2017_2018, df_2018_2019, df_2019_2020 = ["" for _ in range(6)]
df_2020_2021 = ""

# show all row
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)

sc = StandardScaler()
le = LabelEncoder()

knn_classifier = KNeighborsClassifier(n_neighbors=7)
logistic_classifier = LogisticRegression()
nb_classifier = GaussianNB()
clf = tree.DecisionTreeClassifier(criterion='entropy')
random_forest_classifier = RandomForestClassifier(n_estimators=9, max_depth=5, random_state=1, criterion='entropy')
lda_classifier = LDA()
qda_classifier = QDA()
svm_classifier = svm.SVC(kernel='linear')

"""
FTHG = Full Time Home Team Goals
FTAG = Full Time Away Team Goals
FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
HTHG = Half Time Home Team Goals
HTAG = Half Time Away Team Goals
HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
"""

# score_attributes = ['FTHG', 'FTAG', 'FTR', 'HTHG' 'HTAG', 'HTR']
score_attributes = ['HomeTeam_Digit', 'AwayTeam_Digit', 'FTHG', 'FTAG', 'HomeWinPercentage', 'AwayWinPercentage']
score_attributes_test = ['HomeTeam_Digit', 'AwayTeam_Digit', 'FTHG', 'FTAG', 'HomeWinPercentage', 'AwayWinPercentage']

# try to predict score
pred_score_attributes = ['HomeTeam_Digit', 'AwayTeam_Digit', 'HomeWinPercentage', 'AwayWinPercentage', 'Winner_Label']
pred_score_y_hg = 'FTHG'
pred_score_y_ag = 'FTAG'

pred_score = ['FTHG_pred', 'FTAG_pred', 'Winner_Label']
teams_diff = {}

y_test = [0, 0, 0]
y_test_hg = []
y_test_ag = []

output_dir = r'plot'


def load_all_datasets():
    global df_2014_2015, df_2015_2016, df_2016_2017, df_2017_2018, df_2018_2019, df_2019_2020
    df_2014_2015 = pd.read_csv('Datasets/2014-15.csv')
    df_2015_2016 = pd.read_csv('Datasets/2015-16.csv')
    df_2016_2017 = pd.read_csv('Datasets/2016-17.csv')
    df_2017_2018 = pd.read_csv('Datasets/2017-18.csv')
    df_2018_2019 = pd.read_csv('Datasets/2018-19.csv')
    df_2019_2020 = pd.read_csv('Datasets/2019-20.csv')

    df_2014_2015['Winner'] = df_2014_2015.apply(lambda row: label_match_winner(row), axis=1)
    df_2014_2015['Winner_Label'] = df_2014_2015.apply(lambda row: label_match_winner_digit(row), axis=1)
    df_2015_2016['Winner'] = df_2015_2016.apply(lambda row: label_match_winner(row), axis=1)
    df_2015_2016['Winner_Label'] = df_2015_2016.apply(lambda row: label_match_winner_digit(row), axis=1)
    df_2016_2017['Winner'] = df_2016_2017.apply(lambda row: label_match_winner(row), axis=1)
    df_2016_2017['Winner_Label'] = df_2016_2017.apply(lambda row: label_match_winner_digit(row), axis=1)
    df_2017_2018['Winner'] = df_2017_2018.apply(lambda row: label_match_winner(row), axis=1)
    df_2017_2018['Winner_Label'] = df_2017_2018.apply(lambda row: label_match_winner_digit(row), axis=1)
    df_2018_2019['Winner'] = df_2018_2019.apply(lambda row: label_match_winner(row), axis=1)
    df_2018_2019['Winner_Label'] = df_2018_2019.apply(lambda row: label_match_winner_digit(row), axis=1)
    df_2019_2020['Winner'] = df_2019_2020.apply(lambda row: label_match_winner(row), axis=1)
    df_2019_2020['Winner_Label'] = df_2019_2020.apply(lambda row: label_match_winner_digit(row), axis=1)

    define_team_digits(df_2014_2015)
    define_team_digits(df_2015_2016)
    define_team_digits(df_2016_2017)
    define_team_digits(df_2017_2018)
    define_team_digits(df_2018_2019)
    define_team_digits(df_2019_2020)


def define_team_digits(df):
    global teams_diff
    team = df['HomeTeam'].to_dict()

    add_team_define(team)

    df['HomeTeam_Digit'] = df['HomeTeam'].map(teams_diff)
    df['AwayTeam_Digit'] = df['AwayTeam'].map(teams_diff)


def label_home_score(row):
    result = row['Result']
    if isinstance(result, str):
        if len(result) > 0:
            results = result.split(' - ')
            return int(results[0])
        else:
            return 0
    else:
        return 0


def label_away_score(row):
    result = row['Result']
    if isinstance(result, str):
        if len(result) > 0:
            results = result.split(' - ')
            return int(results[1])
        else:
            return 0
    else:
        return 0


def label_winner(row):
    result = row['Result']
    if isinstance(result, str):
        if len(result) > 0:
            results = result.split(' - ')
            if results[0] == results[1]:
                return 'D'
            elif results[0] > results[1]:
                return 'H'
            else:
                return 'A'
        else:
            return ''
    else:
        return ''


def label_pred_winner(row):
    home_pre_score = row['FTHG_pred']
    away_pre_score = row['FTAG_pred']

    if home_pre_score > away_pre_score:
        return 'H'
    elif away_pre_score > home_pre_score:
        return 'A'
    else:
        return 'D'


def load_prediction_dataset():
    global df_2020_2021, y_test
    df_2020_2021 = pd.read_csv('epl-2020.csv')
    df_2020_2021 = df_2020_2021.rename(columns={'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam'})

    # Labelling to be the same dataset
    df_2020_2021['FTHG'] = df_2020_2021.apply(lambda row: label_home_score(row), axis=1)
    df_2020_2021['FTAG'] = df_2020_2021.apply(lambda row: label_away_score(row), axis=1)
    df_2020_2021['FTR'] = df_2020_2021.apply(lambda row: label_winner(row), axis=1)
    df_2020_2021['Winner'] = df_2020_2021.apply(lambda row: label_match_winner(row), axis=1)
    df_2020_2021['Winner_Label'] = df_2020_2021.apply(lambda row: label_match_winner_digit(row), axis=1)

    define_team_digits(df_2020_2021)

    y_test[0] = df_2020_2021['Winner_Label'].values
    y_test[1] = df_2020_2021[pred_score_y_hg].values
    y_test[2] = df_2020_2021[pred_score_y_ag].values


def label_match_winner(row):
    ft_result = row['FTR']
    if ft_result == 'H':
        return row['HomeTeam']
    elif ft_result == 'A':
        return row['AwayTeam']
    else:
        return None


def label_match_winner_digit(row):
    ft_result = row['FTR']
    if ft_result == 'H':
        return 1
    elif ft_result == 'A':
        return 2
    elif ft_result == 'D':
        return 3
    else:
        return 0


def add_team_define(team_dict):
    for key, value in team_dict.items():
        if value in teams_diff.keys():
            continue
        else:
            teams_diff[value] = len(teams_diff) + 1


def update_team_wins(team_wins, season_wins, home=False, away=False):
    for key, value in season_wins.items():
        if home or away:
            if key[0] == key[1]:
                team = key[0]
            else:
                continue
        else:
            team = key
        if team in team_wins:
            team_wins[team] += value
        else:
            team_wins[team] = value


def get_number_of_matches(df, team, home=False, away=False):
    if home:
        return df[df['HomeTeam'] == team].shape[0]
    elif away:
        return df[df['AwayTeam'] == team].shape[0]
    else:
        return df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].shape[0]


def create_df_for_winner_count_percentage(team_wins, home=False, away=False):
    data_to_add = []
    for key, value in team_wins.items():
        numberOfMatchesByTeam = 0
        numberOfMatchesByTeam += get_number_of_matches(df_2014_2015, key, home=home, away=away)
        numberOfMatchesByTeam += get_number_of_matches(df_2015_2016, key, home=home, away=away)
        numberOfMatchesByTeam += get_number_of_matches(df_2016_2017, key, home=home, away=away)
        numberOfMatchesByTeam += get_number_of_matches(df_2017_2018, key, home=home, away=away)
        numberOfMatchesByTeam += get_number_of_matches(df_2018_2019, key, home=home, away=away)
        numberOfMatchesByTeam += get_number_of_matches(df_2019_2020, key, home=home, away=away)
        numberOfWin = team_wins[key]
        team = key
        data_to_add.append(
            {'Team': team, 'NumberOfWin': numberOfWin, 'WinPercentage': (numberOfWin / numberOfMatchesByTeam)})

    df_winner_count_percentage = pd.DataFrame(data_to_add, columns=['Team', 'NumberOfWin', 'WinPercentage'])
    return df_winner_count_percentage


def find_which_team_has_the_most_wins(home=False, away=False):
    global df_2014_2015, df_2015_2016, df_2016_2017, df_2017_2018, df_2018_2019, df_2019_2020

    if home:
        groupby = ['HomeTeam', 'Winner']
    elif away:
        groupby = ['AwayTeam', 'Winner']
    else:
        groupby = 'Winner'

    team_wins = {}
    team_wins_2014_2015 = df_2014_2015.groupby(groupby).count()
    update_team_wins(team_wins, team_wins_2014_2015['Div'].sort_values(ascending=False).to_dict(), home=home, away=away)
    team_wins_2015_2016 = df_2015_2016.groupby(groupby).count()
    update_team_wins(team_wins, team_wins_2015_2016['Div'].sort_values(ascending=False).to_dict(), home=home, away=away)
    team_wins_2016_2017 = df_2016_2017.groupby(groupby).count()
    update_team_wins(team_wins, team_wins_2016_2017['Div'].sort_values(ascending=False).to_dict(), home=home, away=away)
    team_wins_2017_2018 = df_2017_2018.groupby(groupby).count()
    update_team_wins(team_wins, team_wins_2017_2018['Div'].sort_values(ascending=False).to_dict(), home=home, away=away)
    team_wins_2018_2019 = df_2018_2019.groupby(groupby).count()
    update_team_wins(team_wins, team_wins_2018_2019['Div'].sort_values(ascending=False).to_dict(), home=home, away=away)
    team_wins_2019_2020 = df_2018_2019.groupby(groupby).count()
    update_team_wins(team_wins, team_wins_2019_2020['Div'].sort_values(ascending=False).to_dict(), home=home, away=away)

    return create_df_for_winner_count_percentage(team_wins, home=home, away=away)


def find_which_team_has_the_most_wins_at_home():
    return find_which_team_has_the_most_wins(home=True)


def find_which_team_has_the_most_wins_at_away():
    return find_which_team_has_the_most_wins(away=True)


def check_accurarcy(predict, name, i):
    accuracy = np.mean(predict == y_test[i])
    print("Accuracy of", name, "for is", round(float(accuracy), 3))
    rms = mean_squared_error(y_test[i], predict, squared=False)
    return round(float(accuracy), 3), rms


def knn_predict(x, y, x_test, i):
    knn_classifier.fit(x, y)
    predicted = knn_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'kNN', i)


def logistic_predict(x, y, x_test, i):
    logistic_classifier.fit(x, y)
    predicted = logistic_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'logistic regression', i)


def naive(x, y, x_test, i):
    nb_classifier.fit(x, y)
    predicted = nb_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'naive bayesian', i)


def tree(x, y, x_test, i):
    clf.fit(x, y)
    predicted = clf.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'decision tree', i)


def forest(x, y, x_test, i):
    random_forest_classifier.fit(x, y)
    predicted = random_forest_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'random forest', i)


def lda(x, y, x_test, i):
    lda_classifier.fit(x, y)
    predicted = lda_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'LDA', i)


def qda(x, y, x_test, i):
    qda_classifier.fit(x, y)
    predicted = qda_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'QDA', i)


def svm(x, y, x_test, i):
    svm_classifier.fit(x, y)
    predicted = svm_classifier.predict(np.asmatrix(x_test))
    return check_accurarcy(predicted, 'SVM', i)


def apply_winner_label_to_df(df, sorted_df_home_winner, sorted_df_away_winner):
    home_win_percent = pd.Series(sorted_df_home_winner.WinPercentage.values, index=sorted_df_home_winner.Team).to_dict()
    away_win_percent = pd.Series(sorted_df_away_winner.WinPercentage.values, index=sorted_df_away_winner.Team).to_dict()

    df['HomeWinPercentage'] = df['HomeTeam'].map(home_win_percent)
    df['AwayWinPercentage'] = df['AwayTeam'].map(away_win_percent)


def draw(accuracy_rmse, title, i):
    plt.figure(i)
    accuracy = []
    rmse = []
    for data in accuracy_rmse:
        accuracy.append(data[0])
        rmse.append(data[1])

    method = []
    if i == 0:
        method = ['kNN', 'Logistic', 'Naive', 'Decision Tree', 'Random Forest', 'LDA', 'QDA', 'SVM']
    elif i == 1 or 2:
        method = ['kNN', 'Logistic', 'Naive', 'Decision Tree', 'Random Forest', 'LDA', 'SVM']

    plt.bar(method, accuracy, color='blue')
    plt.title(f'Accuracy of all model for {title}')
    plt.xlabel('Prediction Model')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(os.getcwd(), output_dir, 'accuracy_epl_' + title + '.png'))
    plt.show()

    plt.figure(i+1)
    plt.bar(method, rmse, color='blue')
    plt.title(f'Root mean squared for {title}')
    plt.xlabel('Prediction Model')
    plt.ylabel('RMSE')
    plt.savefig(os.path.join(os.getcwd(), output_dir, 'rmse_epl_' + title + '.png'))
    plt.show()


def predicting_champion(i):
    title = 'match_score_home_prediction' if i == 1 else 'match_score_away_prediction'
    y_train_pre = pred_score_y_hg if i == 1 else pred_score_y_ag
    global accuracy
    x1_pred_score = df_2014_2015[pred_score_attributes].values
    x2_pred_score = df_2015_2016[pred_score_attributes].values
    x3_pred_score = df_2016_2017[pred_score_attributes].values
    x4_pred_score = df_2017_2018[pred_score_attributes].values
    x5_pred_score = df_2018_2019[pred_score_attributes].values
    x6_pred_score = df_2019_2020[pred_score_attributes].values
    x_trained_2 = np.concatenate(
        [x1_pred_score, x2_pred_score, x3_pred_score, x4_pred_score, x5_pred_score, x6_pred_score])
    y1_pred_score = df_2014_2015[y_train_pre].values
    y2_pred_score = df_2015_2016[y_train_pre].values
    y3_pred_score = df_2016_2017[y_train_pre].values
    y4_pred_score = df_2017_2018[y_train_pre].values
    y5_pred_score = df_2018_2019[y_train_pre].values
    y6_pred_score = df_2019_2020[y_train_pre].values
    y_trained_2 = np.concatenate(
        [y1_pred_score, y2_pred_score, y3_pred_score, y4_pred_score, y5_pred_score, y6_pred_score])
    x_test_2 = df_2020_2021[pred_score_attributes].values

    random_forest_classifier.fit(x_trained_2, y_trained_2)
    predicted = random_forest_classifier.predict(np.asmatrix(x_test_2))

    return predicted


def predicting_match_score(i):
    title = 'match_score_home_prediction' if i == 1 else 'match_score_away_prediction'
    y_train_pre = pred_score_y_hg if i == 1 else pred_score_y_ag
    global accuracy
    x1_pred_score = df_2014_2015[pred_score_attributes].values
    x2_pred_score = df_2015_2016[pred_score_attributes].values
    x3_pred_score = df_2016_2017[pred_score_attributes].values
    x4_pred_score = df_2017_2018[pred_score_attributes].values
    x5_pred_score = df_2018_2019[pred_score_attributes].values
    x6_pred_score = df_2019_2020[pred_score_attributes].values
    x_trained_2 = np.concatenate(
        [x1_pred_score, x2_pred_score, x3_pred_score, x4_pred_score, x5_pred_score, x6_pred_score])
    y1_pred_score = df_2014_2015[y_train_pre].values
    y2_pred_score = df_2015_2016[y_train_pre].values
    y3_pred_score = df_2016_2017[y_train_pre].values
    y4_pred_score = df_2017_2018[y_train_pre].values
    y5_pred_score = df_2018_2019[y_train_pre].values
    y6_pred_score = df_2019_2020[y_train_pre].values
    y_trained_2 = np.concatenate(
        [y1_pred_score, y2_pred_score, y3_pred_score, y4_pred_score, y5_pred_score, y6_pred_score])
    x_test_2 = df_2020_2021[pred_score_attributes].values
    accuracy = [knn_predict(x_trained_2, y_trained_2, x_test_2, i),
                logistic_predict(x_trained_2, y_trained_2, x_test_2, i),
                naive(x_trained_2, y_trained_2, x_test_2, i), tree(x_trained_2, y_trained_2, x_test_2, i),
                forest(x_trained_2, y_trained_2, x_test_2, i), lda(x_trained_2, y_trained_2, x_test_2, i),
                svm(x_trained_2, y_trained_2, x_test_2, i)]
    draw(accuracy, title, i)


def label_who_win(row):
    ft_result = row['FTR_pred']
    if ft_result == 'H':
        return row['HomeTeam']
    elif ft_result == 'A':
        return row['AwayTeam']
    else:
        return None


if __name__ == '__main__':
    load_all_datasets()
    load_prediction_dataset()
    # Which team is the strongest of the the years? (most wins)
    df_all_winner_percentage = find_which_team_has_the_most_wins()
    sorted_df_winner = df_all_winner_percentage.sort_values('NumberOfWin', ascending=False)

    print(
        f'The team had have most wins in the last 6 seasons is: {sorted_df_winner.iloc[0]["Team"]} with {sorted_df_winner.iloc[0]["NumberOfWin"]}')

    # Who is the strongest home team over the the years? (most wins at home)
    df_all_home_winner_percentage = find_which_team_has_the_most_wins_at_home()
    sorted_df_home_winner = df_all_home_winner_percentage.sort_values('NumberOfWin', ascending=False)

    print(
        f'The team had have most wins at home in the last 6 seasons is: {sorted_df_home_winner.iloc[0]["Team"]} with {sorted_df_home_winner.iloc[0]["NumberOfWin"]}')

    # Who is the strongest away team over the the years? (most wins at away)
    df_all_away_winner_percentage = find_which_team_has_the_most_wins_at_away()
    sorted_df_away_winner = df_all_away_winner_percentage.sort_values('NumberOfWin', ascending=False)
    print(
        f'The team had have most wins when away in the last 6 seasons is: {sorted_df_away_winner.iloc[0]["Team"]} with {sorted_df_away_winner.iloc[0]["NumberOfWin"]}')

    apply_winner_label_to_df(df_2014_2015, sorted_df_home_winner, df_all_away_winner_percentage)
    apply_winner_label_to_df(df_2015_2016, sorted_df_home_winner, df_all_away_winner_percentage)
    apply_winner_label_to_df(df_2016_2017, sorted_df_home_winner, df_all_away_winner_percentage)
    apply_winner_label_to_df(df_2017_2018, sorted_df_home_winner, df_all_away_winner_percentage)
    apply_winner_label_to_df(df_2018_2019, sorted_df_home_winner, df_all_away_winner_percentage)
    apply_winner_label_to_df(df_2019_2020, sorted_df_home_winner, df_all_away_winner_percentage)
    apply_winner_label_to_df(df_2020_2021, sorted_df_home_winner, df_all_away_winner_percentage)

    df_2014_2015['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2014_2015['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)
    df_2015_2016['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2015_2016['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)
    df_2016_2017['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2016_2017['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)
    df_2017_2018['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2017_2018['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)
    df_2018_2019['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2018_2019['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)
    df_2019_2020['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2019_2020['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)
    df_2020_2021['HomeWinPercentage'] = df_2020_2021['HomeWinPercentage'].fillna(0)
    df_2020_2021['AwayWinPercentage'] = df_2020_2021['AwayWinPercentage'].fillna(0)

    # is it possible to build a classifier to identify big team?
    x1 = df_2014_2015[score_attributes].values
    x2 = df_2015_2016[score_attributes].values
    x3 = df_2016_2017[score_attributes].values
    x4 = df_2017_2018[score_attributes].values
    x5 = df_2018_2019[score_attributes].values
    x6 = df_2019_2020[score_attributes].values

    x_trained = np.concatenate([x1, x2, x3, x4, x5, x6])

    y1 = df_2014_2015['Winner_Label'].values
    y2 = df_2015_2016['Winner_Label'].values
    y3 = df_2016_2017['Winner_Label'].values
    y4 = df_2017_2018['Winner_Label'].values
    y5 = df_2018_2019['Winner_Label'].values
    y6 = df_2019_2020['Winner_Label'].values

    y_trained = np.concatenate([y1, y2, y3, y4, y5, y6])

    x_test = df_2020_2021[score_attributes_test].values
    accuracy = [knn_predict(x_trained, y_trained, x_test, 0), logistic_predict(x_trained, y_trained, x_test, 0),
                naive(x_trained, y_trained, x_test, 0), tree(x_trained, y_trained, x_test, 0),
                forest(x_trained, y_trained, x_test, 0), lda(x_trained, y_trained, x_test, 0),
                qda(x_trained, y_trained, x_test, 0), svm(x_trained, y_trained, x_test, 0)]
    draw(accuracy, 'match_winner_prediction', 0)

    # predicting result and champion for current season 20/21?
    predicting_match_score(1)
    predicting_match_score(2)

    # Who is the champion of this season
    # pick random forest for both home and away score.
    pred_home_score = predicting_champion(1)
    pred_away_score = predicting_champion(2)

    df_2020_2021['FTHG_pred'] = pd.Series(pred_home_score)
    df_2020_2021['FTAG_pred'] = pd.Series(pred_away_score)
    df_2020_2021['FTR_pred'] = df_2020_2021.apply(lambda row: label_pred_winner(row), axis=1)
    df_2020_2021['Winner_pred'] = df_2020_2021.apply(lambda row: label_who_win(row), axis=1)

    winner_teams = df_2020_2021.groupby('Winner_pred').count()['Date'].sort_values(ascending=False)
    print(f'From the prediction the winner team will be {winner_teams.index[0]} of season 2020/21 with number of wins {winner_teams.iloc[0]}')



