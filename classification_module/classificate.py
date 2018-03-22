
from classification_module.prepare_ratio import get_model_ratio
from classification_module.numpy_proceed import preprocess_array
from doc2vec_module.train_model import get_model_for_genre
from doc2vec_module.constants import FileConstants
from sklearn.multiclass import OneVsRestClassifier
from gensim.models import Doc2Vec
from doc2vec_module import load
from pathlib import Path
from sklearn import svm
import numpy as np


# Пример работы для отдельных книг
random_state = np.random.RandomState(6)
lin_clf = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))


path_to_check1 = Path(__file__).parents[1].joinpath('путь ко второй книге')
documents1 = load.get_doc_from_file(str(path_to_check1))

path_to_check2 = Path(__file__).parents[1].joinpath('путь к первой книге)
#documents2 = load.get_doc_from_file(str(path_to_check2))

print("Получаем модель для книги")
print("")
check_model1 = get_model_for_genre([documents1])
#heck_model2 = get_model_for_genre([documents2])

print("Начинаем подгружать модели по жанрам")
print(" ")

prop_list = ["MODEL_ADVENTURE", "MODEL_ART", "MODEL_DETECTIVE", "MODEL_FANTASTIC", "MODEL_FANTASY", "MODEL_LOVE"]
genre_labels = ['приключения', 'искусство', 'детектив', 'фантастика', 'фэнтези', 'любовь']
train_list = []

for item in prop_list:
    print("Модель: {0}".format(item))
    model_prop = getattr(FileConstants, item)
    model = Doc2Vec.load(model_prop.fget(FileConstants()))
    train = preprocess_array(get_model_ratio(model))
    train_list.append(train)
    print(" ")

check_train1 = preprocess_array(np.array(check_model1.docvecs[str(0)]))
np_train_list = np.asarray(train_list)

counter = 0
counter1 = 0
check = [1, 2, 3, 4, 5, 6]

lin_clf.fit(np_train_list, check)

test = []

for i in range(6):
    test.append(np.random.randint(200, size = 20))

test = np.asarray(test)

for i in range(6):
    print("Расчёт для жанра \"{0}\"".format(genre_labels[counter]))
    test[counter] = check_train1

    arr = lin_clf.predict(test)

    result = (np.sum(arr) / np.size(arr)) / 10

    print(lin_clf.predict(test))
    counter +=1
