def create_features(df):

    df["player_diff"] = df["players_team_a"] - df["players_team_b"]

    df["ball_zone"] = df["ball_x"] // 100

    return df