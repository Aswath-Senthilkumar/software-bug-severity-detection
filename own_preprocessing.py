import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Bug:
    def __init__(self, project_name: str, project_version: str, severity: int, code: str, code_comment: str,
                 code_no_comment: str, lc: int, pi: int, ma: int, nbd: int, ml: int, d: int, mi: int, fo: int,
                 r: int, e: int):
        self.project_name = project_name
        self.project_version = project_version
        self.label = severity
        self.code = code
        self.code_comment = code_comment
        self.code_no_comment = code_no_comment
        self.lc = lc
        self.pi = pi
        self.ma = ma
        self.nbd = nbd
        self.ml = ml
        self.d = d
        self.mi = mi
        self.fo = fo
        self.r = r
        self.e = e


def split_dataset(d4j_path: str, bugs_jar_path: str):
    logging.info("Reading input data...")
    d4j = pd.read_csv(d4j_path)
    new_d4j = convert_severity_to_numeric(d4j, d4j_mapping=True)
    bugs_jar = pd.read_csv(bugs_jar_path)
    new_bugs_jar = convert_severity_to_numeric(bugs_jar, d4j_mapping=False)

    bugs = create_bugs(new_d4j) + create_bugs(new_bugs_jar)

    df_bugs = pd.DataFrame(bugs)
    df_bugs.drop_duplicates(keep='first', inplace=True)

    train, test = train_test_split(df_bugs, test_size=0.15, random_state=666, shuffle=True, stratify=df_bugs['label'])
    train, val = train_test_split(train, test_size=0.15, random_state=666, shuffle=True, stratify=train['label'])

    cols = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']
    scaler = RobustScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    val[cols] = scaler.transform(val[cols])

    write_bugs(train, "train_scaled")
    write_bugs(val, "valid_scaled")
    write_bugs(test, "test_scaled")


def create_bugs(df: pd.DataFrame) -> list:
    return df[df["IsBuggy"]].apply(lambda row: Bug(
        project_name=row["ProjectName"],
        project_version=row["ProjectVersion"],
        severity=row["Severity"],
        code=row["SourceCode"],
        code_comment=row["CodeComment"],
        code_no_comment=row["CodeNoComment"],
        lc=row["LC"],
        pi=row["PI"],
        ma=row["MA"],
        nbd=row["NBD"],
        ml=row["ML"],
        d=row["D"],
        mi=row["MI"],
        fo=row["FO"],
        r=row["R"],
        e=row["E"]
    ).__dict__, axis=1).tolist()


def write_bugs(bugs: pd.DataFrame, name: str):
    logging.info(f"Writing {name} data to files...")
    with open(f"{name}.jsonl", 'w') as f:
        for bug in bugs.to_dict("records"):
            f.write(json.dumps(bug) + "\n")
    bugs.to_csv(f"{name}.csv", index=False)


def convert_severity_to_numeric(df: pd.DataFrame, d4j_mapping: bool = True) -> pd.DataFrame:
    severity_mapping = {
        'Critical': 0,
        'High': 1,
        'Medium': 2,
        'Low': 3
    } if d4j_mapping else {
        'Blocker': 0,
        'Critical': 0,
        'Major': 1,
        'Minor': 3,
        'Trivial': 3
    }
    df['Severity'] = df['Severity'].map(severity_mapping).fillna(3)
    return df


if __name__ == '__main__':
    split_dataset('/Users/aswath/Downloads/Fall 2024/DM/Final Project/Code/d4j_methods_sc_metrics_comments.csv', '/Users/aswath/Downloads/Fall 2024/DM/Final Project/Code/bugsjar_methods_sc_metrics_comments.csv')