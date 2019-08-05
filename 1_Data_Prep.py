data_list = list()
columns = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']

# inspired by https://thedatafrog.com/text-mining-pandas-yelp/
with open('review.json') as reviews:
    
    import json
    for i, line in enumerate(reviews):

        # convert json line to dict
        data = json.loads(line)
        data_list.append([data['review_id'], data['user_id'],
                          data['business_id'], data['stars'],
                          data['useful'], data['funny'],
                          data['cool'], data['text'], data['date']])

reviews.close()

import pandas
df = pandas.DataFrame(data_list, columns=columns)

df_sample = df.sample(frac=1).reset_index(drop=True).head(100000)

df_sample.to_csv('shuffled.100000.reviews.csv')
df_sample.to_hdf('reviews_100000.h5', key='df', mode='w')
