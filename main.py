from src.features.build_features import rider_features

# 'date', 'distance', 'result ranking'
features = rider_features(30)

if features is None:
    print('does not return')
else:
    dates = features[:, 0]
    dists = features[:, 1]
    ranks = features[:, -1]