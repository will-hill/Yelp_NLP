data_list = list()
columns = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']

# inspired by https://thedatafrog.com/text-mining-pandas-yelp/
with open('review.json') as reviews:
    import json

    REVIEWS_TO_INGEST = 1000
    for i, line in enumerate(reviews):

        if i == REVIEWS_TO_INGEST:
            break

            # convert json line to di t
        data = json.loads(line)
        data_list.append([data['review_id'],
                          data['user_id'],
                          data['business_id'],
                          data['stars'],
                          data['useful'],
                          data['funny'],
                          data['cool'],
                          data['text'],
                          data['date']])

reviews.close()
del reviews, i, line, data, REVIEWS_TO_INGEST
###
import pandas
df = pandas.DataFrame(data_list, columns=columns)
del data_list, columns, pandas

