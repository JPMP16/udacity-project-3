import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
import pickle
from starter.ml.data import process_data

df = pd.read_csv('data/census.csv')
with open('model/rf.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
label = df['salary']
df.drop('salary', axis=1, inplace=True)
input_data, _, _, _ = process_data(df, categorical_features=cat_features, lb=lb, encoder=encoder, training=False)
score = model.predict_proba(input_data)[:,1]

df['score'] = score
df['label_value'] = label

keep_cols = [
        "workclass",
        "education",
        "marital-status",
        "race",
        "sex",
        "score",
        "label_value"
    ]
df = df[keep_cols]

group = Group()
xtab, _ = group.get_crosstabs(df)

bias = Bias()
bias_df = bias.get_disparity_predefined_groups(xtab,
                                               original_df=df,
                                               ref_groups_dict={"race": "White", "sex": "Male", "marital-status": "Never-married", "workclass": 'Private', 'education': 'HS-grad'},
                                               alpha=0.05,
                                               mask_significance=True)
cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
fairness = Fairness()
fairness_df = fairness.get_group_value_fairness(bias_df)

overall_fairness = fairness.get_overall_fairness(fairness_df)
print(overall_fairness)