import pandas as pd

def build_dataset(result_json):

    rows = []

    for frame in result_json["frames"]:

        row = {
            "ball_x": frame["ball"][0],
            "ball_y": frame["ball"][1],
            "players_team_a": frame["players_team_a"],
            "players_team_b": frame["players_team_b"],
            "possession": frame["possession"]
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df