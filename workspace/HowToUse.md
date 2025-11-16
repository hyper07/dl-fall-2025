```python
import psycopg, os
from PIL import Image
from IPython.display import display
import pandas as pd
from tabulate import tabulate
from pymongo import MongoClient
```

---

# FOR MongoDB


```python
from pymongo import MongoClient
client = MongoClient('mongodb://admin:PassW0rd@apan-mongo:27017/')
```


```python
dbnames = client.list_database_names()
dbnames
```




    ['admin', 'config', 'local']




```python
db = client.test
```


```python
collection = db.test
```


```python
collection
```




    Collection(Database(MongoClient(host=['apan-mongo:27017'], document_class=dict, tz_aware=False, connect=True), 'test'), 'test')




```python
#collection = db.test

import datetime 
from datetime import datetime

post = {'singer': 'Louis Armstrong',
        "song": "What a wonderful world",
        "tags":["jazz", "blues"],
        "date": datetime.now()
}

post_id = collection.insert_one(post).inserted_id
```


```python
print('Our first post id: {0}'.format(post_id))
print('Our first post: {0}'.format(post))
```

    Our first post id: 67cc5ab298c392a2f6b3d821
    Our first post: {'singer': 'Louis Armstrong', 'song': 'What a wonderful world', 'tags': ['jazz', 'blues'], 'date': datetime.datetime(2025, 3, 8, 14, 56, 50, 401266), '_id': ObjectId('67cc5ab298c392a2f6b3d821')}



```python
collection.drop()
```


```python
client.close()
```


```python

```


```python

```


```python

```




```python

```


```python

```


```python

```
