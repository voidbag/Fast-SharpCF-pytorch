import os
from urllib import request
from tqdm.auto import tqdm

if __name__ == "__main__":
    url_dir = "https://raw.githubusercontent.com/hexiangnan/adversarial_personalized_ranking/54f708f9fe4338cdb5ffe82205e69327bf22466a/Data"
    li_fname = ["ml-1m.test.negative", "ml-1m.test.rating", "ml-1m.train.rating",
                "pinterest-20.test.negative", "pinterest-20.test.rating", "pinterest-20.train.rating",
                "yelp.test.negative", "yelp.test.rating", "yelp.train.rating"]
    dir_data = "./Data" 
    os.makedirs(dir_data, exist_ok=True)
    for fname in tqdm(li_fname, total=len(li_fname)):
        url = f"{url_dir}/{fname}"
        print(url)
        data = request.urlopen(url).read()
        with open(os.path.join(dir_data, fname), "wb") as f:
            f.write(data)
