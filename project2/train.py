import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 전처리
def make_features(hand):
    ranks = [] # 카드의 숫자 
    suits = [] # 카드의 문양

    for suit, rank in hand: # hand 원소가 튜플 형태(문양, 숫자)로 들어오게 하기때문에
        ranks.append(rank)
        suits.append(suit)

    ranks.sort() 

    # 같은 숫자와 무늬가 각각 최대 몇 개 있는지 확인
    max_rank = max(ranks.count(r) for r in ranks)
    max_suit = max(suits.count(s) for s in suits)

    # 페어가 몇 개 있는지 확인
    pair_count = 0

    for rank in set(ranks): 
        if ranks.count(rank) == 2:
            pair_count += 1

    return ranks + [max_rank, max_suit, pair_count] # 데이터를 숫자,같은 숫자, 같은 문양이 몇개있는지 , 페어 수 로 데이터 정보 추가


# 학습 데이터 불러오기
data = pd.read_csv('poker-hand-training.csv').to_numpy()

# 정답 레이블
y = data[:, 10]


# 3장 모델과 4장 모델 각각 학습
for card_count in [3, 4]:

    X = []

    for row in data: 
        hand = []

        # 공개된 카드만 가져오기
        for i in range(card_count):
            suit = row[i * 2] # 패가 3장일때 문양 인덱스가 0, 2, 4이고 4장일때 0,2,4,6이므로
            rank = row[i * 2 + 1] # 숫자 인덱스가 1,3,5 / 1,3,5,7이므로
            hand.append((suit, rank)) # 튜플 형태(문양, 숫자)로 저장

        X.append(make_features(hand)) # 전처리


    # 랜덤 포레스트 모델 생성
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42
    )

    # 학습
    model.fit(X, y)

    # 모델 저장
    pickle.dump(model, open(f'poker-model{card_count}.sav', 'wb'))
