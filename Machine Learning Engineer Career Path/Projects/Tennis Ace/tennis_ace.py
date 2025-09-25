import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis = pd.read_csv('tennis_stats.csv')
print(tennis.columns)
print(tennis.info())

# perform exploratory analysis here:
offensive_cols = ['Aces', 'DoubleFaults', 'FirstServe', 'FirstServePointsWon', 'SecondServePointsWon',
                  'BreakPointsFaced', 'BreakPointsSaved', 'ServiceGamesPlayed', 'ServiceGamesWon',
                  'TotalServicePointsWon']

defensive_cols = ['FirstServeReturnPointsWon', 'SecondServeReturnPointsWon', 'BreakPointsOpportunities',
                  'BreakPointsConverted', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon', 'TotalPointsWon']

outcome_cols = ['Wins', 'Losses', 'Winnings', 'Ranking']


# Good positive correlation
# plt.scatter(tennis[['BreakPointsOpportunities']], tennis[['Winnings']], alpha=.4)
# plt.show()

# No correlation
# plt.scatter(tennis[['FirstServeReturnPointsWon']], tennis[['Ranking']], alpha=.4)
# plt.show()

# Very strong positive correlation
# plt.scatter(tennis[['ServiceGamesPlayed']], tennis[['Wins']], alpha=.4)
# plt.show()

# -----No point has good correlation with Ranking-----
# plt.scatter(tennis[['TotalPointsWon']], tennis[['Ranking']], alpha=.4)
# plt.show()

# Function to plot the relationship between a feature and a target
def plot_data(feature_col: str, target_col: str) -> None:
    plt.scatter(tennis[[feature_col]], tennis[[target_col]], alpha=.4)
    plt.title(feature_col + ' vs ' + target_col)
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.show()
    plt.close()


# create method for fitting and plotting chosen features and targets
def model_data(feature_cols: list, target_col: list) -> None:
    X = tennis[feature_cols]
    y = tennis[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    line_fitter = LinearRegression()
    line_fitter.fit(X_train, y_train)

    if len(feature_cols) <= 1:
        print('{} vs {} {:.4f}'.format(feature_cols[0], target_col[0], line_fitter.score(X_test, y_test)))
    else:
        print('{} vs {} {:.4f}'.format(' + '.join(feature_cols), target_col[0], line_fitter.score(X_test, y_test)))

    plt.scatter(y_test, line_fitter.predict(X_test), alpha=.4)
    plt.title(' + '.join(feature_cols) + ' vs ' + target_col[0])
    plt.xlabel('y actual')
    plt.ylabel('y predicted')
    plt.show()
    plt.close()


# all offensive_cols vs individual outcome_cols
print('------All Offensive cols vs all Outcome cols------')
offensive_cols_accuracy = []
for outcome in outcome_cols:
    line_fitter = LinearRegression()

    X = tennis[offensive_cols]
    y = tennis[outcome]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    line_fitter = line_fitter.fit(X_train, y_train)
    accuracy = line_fitter.score(X_test, y_test)
    offensive_cols_accuracy.append(accuracy)
    print(outcome, accuracy)

print()
print('------All Defensive cols vs all Outcome cols------')
# all defensive_cols vs individual outcome_cols
defensive_cols_accuracy = []
for outcome in outcome_cols:
    line_fitter = LinearRegression()

    X = tennis[defensive_cols]
    y = tennis[outcome]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    line_fitter = line_fitter.fit(X_train, y_train)
    accuracy = line_fitter.score(X_test, y_test)
    defensive_cols_accuracy.append(accuracy)
    print(outcome, accuracy)

## perform single feature linear regressions here:
print()
print('------Single Feature Linear Regression Models------')
model_data(['ServiceGamesPlayed'], ['Wins'])

model_data(['FirstServeReturnPointsWon'], ['Winnings'])

model_data(['BreakPointsOpportunities'], ['Winnings'])

## perform two feature linear regressions here:
print()
print('------Two Feature Linear Regression Models------')
plot_data('ServiceGamesPlayed', 'Wins')

plot_data('BreakPointsOpportunities', 'Wins')

model_data(['ServiceGamesPlayed', 'BreakPointsOpportunities'], ['Wins'])

plot_data('BreakPointsFaced', 'Losses')

plot_data('BreakPointsOpportunities', 'Losses')

model_data(['BreakPointsFaced', 'BreakPointsOpportunities'], ['Losses'])

plot_data('ReturnGamesPlayed', 'Losses')

plot_data('ServiceGamesPlayed', 'Losses')

model_data(['ServiceGamesPlayed', 'ReturnGamesPlayed'], ['Losses'])

plot_data('ServiceGamesPlayed', 'Losses')

plot_data('DoubleFaults', 'Losses')

model_data(['ServiceGamesPlayed', 'DoubleFaults'], ['Losses'])

## perform multiple feature linear regressions here:
print('------Multi Feature Linear Regression Models------')

# for i in tennis.columns[2:]:
#   plot_data(i, 'Ranking')

# --------Columns vs Wins-----------
# strong correlation: ServiceGamesPlayed, ReturnGamesPlayed, BreakPointsOpportunities, BreakPointsFaced, (Winnings), (Losses)
# small correlation: Aces, DoubleFaults

# Only datapoints with strong correlation and are not candidate for target column
model_data(['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced'], ['Wins'])

# All datapoints with strong correlation
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Winnings', 'Losses'],
    ['Wins'])

# All datapoints that correlate and are not candidate for target column
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Aces', 'DoubleFaults'],
    ['Wins'])

# All datapoints that correlate
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Aces', 'DoubleFaults',
     'Losses', 'Winnings'], ['Wins'])

# --------Columns vs Losses-----------
# strong correlation: BreakPointsFaced, BreakPointsOpportunities, ReturnGamesPlayed, ServiceGamesPlayed
# small correlation: Aces, DoubleFaults, (Wins), (Winnings)

# Only datapoints with strong correlation and are not candidate for target column
model_data(['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced'], ['Losses'])

# All datapoints that correlate and are not candidate for target column
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Aces', 'DoubleFaults'],
    ['Losses'])

# All datapoints that correlate
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Aces', 'DoubleFaults',
     'Wins', 'Winnings'], ['Losses'])

# --------Columns vs Winnings-----------
# strong correlation: BreakPointsFaced, BreakPointsOpportunities, ReturnGamesPlayed, ServiceGamesPlayed, (Wins), (Losses)
# small correlation: Aces, DoubleFaults

# Only datapoints with strong correlation and are not candidate for target column
model_data(['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced'], ['Winnings'])

# All datapoints with strong correlation
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Wins', 'Losses'],
    ['Winnings'])

# All datapoints that correlate and are not candidate for target column
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Aces', 'DoubleFaults'],
    ['Winnings'])

# All datapoints that correlate
model_data(
    ['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'BreakPointsFaced', 'Aces', 'DoubleFaults',
     'Losses', 'Wins'], ['Winnings'])
