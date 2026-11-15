import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

def make_features(hand):
    ranks = [] 
    suits = [] 

    for suit, rank in hand:
        ranks.append(rank)
        suits.append(suit)

    ranks.sort() 

    max_rank = max(ranks.count(r) for r in ranks)
    max_suit = max(suits.count(s) for s in suits)

    pair_count = 0

    for rank in set(ranks):
        if ranks.count(rank) == 2:
            pair_count += 1

    card_count = len(hand) 
    rank_set = set(ranks)
    same_suit = max_suit == card_count

    straight_possible = 0 
    flush_possible = 0 
    stflush_possible = 0 
    rstflush_possible = 0 

    if same_suit: 
        flush_possible = 1

    if len(rank_set) == card_count:
        for i in range(1,11): 
            straight_ranks = list(range(i, i+5))

            if 14 in straight_ranks: 
                straight_ranks.remove(14)
                straight_ranks.append(1)

            if rank_set.issubset(straight_ranks): 
                straight_possible = 1

                if same_suit:
                    if i == 10:
                        rstflush_possible = 1
                    else:
                        stflush_possible = 1

    

    return ranks + [
        max_rank,
        max_suit,
        pair_count,
        straight_possible,
        flush_possible,
        stflush_possible,
        rstflush_possible
    ] 

data = pd.read_csv('poker-hand-training.csv').to_numpy()

y = data[:, 10]

for card_count in [3, 4]:

    X = []

    for row in data: 
        hand = []

        for i in range(card_count):
            suit = row[i * 2] 
            rank = row[i * 2 + 1] 
            hand.append((suit, rank))

        X.append(make_features(hand))

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X, y)
    
    pickle.dump(model, open(f'poker-model{card_count}.sav', 'wb'))
